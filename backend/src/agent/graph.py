import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from openai import OpenAI
from google.genai import Client
from tavily import TavilyClient

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

# Check API keys - at least one must be available
gemini_available = os.getenv("GEMINI_API_KEY") is not None
deepseek_available = os.getenv("ARK_API_KEY") is not None
tavily_available = os.getenv("TAVILY_API_KEY") is not None

if not gemini_available and not deepseek_available:
    raise ValueError("Either GEMINI_API_KEY or ARK_API_KEY must be set")

# Initialize clients if available
genai_client = None
openai_client = None
tavily_client = None

if gemini_available:
    genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

if deepseek_available:
    openai_client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.getenv("ARK_API_KEY")
    )

if tavily_available:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses either Gemini or DeepSeek to create optimized search queries for web research.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Choose model based on provider
    provider = configurable.model_provider
    
    if provider == "deepseek" and deepseek_available:
        # Use DeepSeek
        llm = ChatOpenAI(
            model=configurable.deepseek_query_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("ARK_API_KEY"),
            base_url=configurable.deepseek_api_base_url,
        )
    elif provider == "gemini" and gemini_available:
        # Use Gemini
        llm = ChatGoogleGenerativeAI(
            model=configurable.gemini_query_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        # Fallback to available provider
        if gemini_available:
            llm = ChatGoogleGenerativeAI(
                model=configurable.gemini_query_model,
                temperature=1.0,
                max_retries=2,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        elif deepseek_available:
            llm = ChatOpenAI(
                model=configurable.deepseek_query_model,
                temperature=1.0,
                max_retries=2,
                api_key=os.getenv("ARK_API_KEY"),
                base_url=configurable.deepseek_api_base_url,
            )
        else:
            raise ValueError("No valid model provider available")

    if provider == "deepseek" and deepseek_available:
        # Use direct structured output for DeepSeek
        structured_llm = llm.with_structured_output(SearchQueryList, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def decide_search_mode(state: OverallState):
    """LangGraph routing function that decides whether to use web search or direct answer.
    
    Routes to either web research or direct answer based on the enable_web_search flag.
    
    Args:
        state: Current graph state containing the enable_web_search flag
        
    Returns:
        String indicating the next node ("generate_query" for web search, "direct_answer" for direct)
    """
    if state.get("enable_web_search", True):  # Default to True for backward compatibility
        return "generate_query"
    else:
        return "direct_answer"


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research.

    Uses either Google Search (Gemini) or Tavily (DeepSeek) based on the configured provider.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    configurable = Configuration.from_runnable_config(config)
    provider = configurable.model_provider
    
    if provider == "deepseek" and deepseek_available and tavily_available:
        # Use DeepSeek + Tavily
        return _web_research_deepseek_tavily(state, configurable)
    elif provider == "gemini" and gemini_available:
        # Use Gemini + Google Search
        return _web_research_gemini(state, configurable)
    else:
        # Fallback to available provider
        if gemini_available:
            return _web_research_gemini(state, configurable)
        elif deepseek_available and tavily_available:
            return _web_research_deepseek_tavily(state, configurable)
        else:
            raise ValueError("No valid web search provider available")


def _web_research_gemini(state: WebSearchState, configurable: Configuration) -> OverallState:
    """Web research using Gemini + Google Search."""
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.gemini_query_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def _web_research_deepseek_tavily(state: WebSearchState, configurable: Configuration) -> OverallState:
    """Web research using DeepSeek + Tavily."""
    # First, perform Tavily search
    search_results = tavily_client.search(
        query=state["search_query"],
        search_depth="advanced",
        max_results=5
    )
    
    # Extract search context
    search_context = ""
    sources_gathered = []
    
    for i, result in enumerate(search_results.get("results", [])):
        search_context += f"Source {i+1}: {result.get('title', '')}\n"
        search_context += f"URL: {result.get('url', '')}\n"
        search_context += f"Content: {result.get('content', '')}\n\n"
        
        # Create source entry
        short_url = f"[{state['id']}-{i+1}]"
        sources_gathered.append({
            "title": result.get('title', 'Unknown'),
            "url": result.get('url', ''),
            "snippet": result.get('content', '')[:200] + "...",
            "short_url": short_url,
            "value": result.get('url', ''),
            "label": result.get('title', 'Web Source')
        })
    
    # Now use DeepSeek to analyze and synthesize the search results
    analysis_prompt = f"""Based on the following search results about "{state["search_query"]}", provide a comprehensive analysis and summary:

Search Results:
{search_context}

Please provide:
1. A detailed summary of the key findings
2. Important facts and insights
3. Any relevant context or background information

Format your response clearly and cite the sources using [1], [2], etc. markers."""

    # Use DeepSeek to analyze
    llm = ChatOpenAI(
        model=configurable.deepseek_query_model,
        temperature=0.3,
        max_retries=2,
        api_key=os.getenv("ARK_API_KEY"),
        base_url=configurable.deepseek_api_base_url,
    )
    
    response = llm.invoke(analysis_prompt)
    
    # Replace source markers with short URLs
    modified_text = response.content
    for i, source in enumerate(sources_gathered):
        modified_text = modified_text.replace(f"[{i+1}]", source["short_url"])

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    provider = configurable.model_provider

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    
    # Choose model based on provider
    if provider == "deepseek" and deepseek_available:
        # Use DeepSeek
        llm = ChatOpenAI(
            model=configurable.deepseek_reflection_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("ARK_API_KEY"),
            base_url=configurable.deepseek_api_base_url,
        )
    elif provider == "gemini" and gemini_available:
        # Use Gemini
        llm = ChatGoogleGenerativeAI(
            model=configurable.gemini_reflection_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        # Fallback
        if gemini_available:
            llm = ChatGoogleGenerativeAI(
                model=configurable.gemini_reflection_model,
                temperature=1.0,
                max_retries=2,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        elif deepseek_available:
            llm = ChatOpenAI(
                model=configurable.deepseek_reflection_model,
                temperature=1.0,
                max_retries=2,
                api_key=os.getenv("ARK_API_KEY"),
                base_url=configurable.deepseek_api_base_url,
            )
        else:
            raise ValueError("No valid model provider available")
    
    if provider == "deepseek" and deepseek_available:
        # Use direct structured output for DeepSeek
        structured_llm = llm.with_structured_output(Reflection, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(Reflection)
    
    result = structured_llm.invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def direct_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that provides direct answer without web research.

    Uses the reasoning model to directly answer the user's question based on 
    the model's training data without performing web searches.

    Args:
        state: Current graph state containing the user's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including messages with the direct answer
    """
    configurable = Configuration.from_runnable_config(config)
    provider = configurable.model_provider

    # Format the prompt for direct answer
    current_date = get_current_date()
    user_question = get_research_topic(state["messages"])
    
    direct_answer_prompt = f"""You are a helpful AI assistant. Today's date is {current_date}.

Please provide a comprehensive and accurate answer to the following question based on your knowledge:

Question: {user_question}

Please provide a detailed, well-structured response. If you're uncertain about any facts, please mention that uncertainty. Do not make up information you're not confident about."""

    # Choose model based on provider
    if provider == "deepseek" and deepseek_available:
        # Use DeepSeek
        llm = ChatOpenAI(
            model=configurable.deepseek_answer_model,
            temperature=0.7,
            max_retries=2,
            api_key=os.getenv("ARK_API_KEY"),
            base_url=configurable.deepseek_api_base_url,
        )
    elif provider == "gemini" and gemini_available:
        # Use Gemini
        llm = ChatGoogleGenerativeAI(
            model=configurable.gemini_answer_model,
            temperature=0.7,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        # Fallback
        if gemini_available:
            llm = ChatGoogleGenerativeAI(
                model=configurable.gemini_answer_model,
                temperature=0.7,
                max_retries=2,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        elif deepseek_available:
            llm = ChatOpenAI(
                model=configurable.deepseek_answer_model,
                temperature=0.7,
                max_retries=2,
                api_key=os.getenv("ARK_API_KEY"),
                base_url=configurable.deepseek_api_base_url,
            )
        else:
            raise ValueError("No valid model provider available")

    result = llm.invoke(direct_answer_prompt)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": [],
    }


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    provider = configurable.model_provider

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # Choose model based on provider
    if provider == "deepseek" and deepseek_available:
        # Use DeepSeek
        llm = ChatOpenAI(
            model=configurable.deepseek_answer_model,
            temperature=0,
            max_retries=2,
            api_key=os.getenv("ARK_API_KEY"),
            base_url=configurable.deepseek_api_base_url,
        )
    elif provider == "gemini" and gemini_available:
        # Use Gemini
        llm = ChatGoogleGenerativeAI(
            model=configurable.gemini_answer_model,
            temperature=0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        # Fallback
        if gemini_available:
            llm = ChatGoogleGenerativeAI(
                model=configurable.gemini_answer_model,
                temperature=0,
                max_retries=2,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        elif deepseek_available:
            llm = ChatOpenAI(
                model=configurable.deepseek_answer_model,
                temperature=0,
                max_retries=2,
                api_key=os.getenv("ARK_API_KEY"),
                base_url=configurable.deepseek_api_base_url,
            )
        else:
            raise ValueError("No valid model provider available")

    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("direct_answer", direct_answer)

# Set the entrypoint with routing decision
# This determines whether to use web search or direct answer
builder.add_conditional_edges(START, decide_search_mode, ["generate_query", "direct_answer"])

# Web search path
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

# Direct answer path
builder.add_edge("direct_answer", END)

graph = builder.compile()
