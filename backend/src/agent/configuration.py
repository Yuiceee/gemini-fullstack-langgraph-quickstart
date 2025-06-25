import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    model_provider: str = Field(
        default="deepseek",
        description="The model provider to use: 'gemini' or 'deepseek'"
    )

    # Gemini models
    gemini_query_model: str = Field(
        default="gemini-2.0-flash",
        description="The Gemini model for query generation."
    )

    gemini_reflection_model: str = Field(
        default="gemini-2.5-flash",
        description="The Gemini model for reflection."
    )

    gemini_answer_model: str = Field(
        default="gemini-2.5-pro",
        description="The Gemini model for answer generation."
    )

    # DeepSeek models
    deepseek_query_model: str = Field(
        default="deepseek-r1-250528",
        description="The DeepSeek model for query generation."
    )

    deepseek_reflection_model: str = Field(
        default="deepseek-r1-250528",
        description="The DeepSeek model for reflection."
    )

    deepseek_answer_model: str = Field(
        default="deepseek-r1-250528",
        description="The DeepSeek model for answer generation."
    )

    deepseek_api_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="The base URL for the DeepSeek API."
    )

    # Legacy fields for backward compatibility
    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        description="Legacy field - use model_provider and specific model fields instead."
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        description="Legacy field - use model_provider and specific model fields instead."
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        description="Legacy field - use model_provider and specific model fields instead."
    )

    number_of_initial_queries: int = Field(
        default=3,
        description="The number of initial search queries to generate."
    )

    max_research_loops: int = Field(
        default=2,
        description="The maximum number of research loops to perform."
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {}
        for name in cls.model_fields.keys():
            # Try environment variable first (uppercase)
            env_value = os.environ.get(name.upper())
            if env_value is not None:
                raw_values[name] = env_value
            # Then try configurable (lowercase)
            elif configurable.get(name) is not None:
                raw_values[name] = configurable.get(name)
            # Use default value if available
            else:
                field_info = cls.model_fields[name]
                if hasattr(field_info, 'default') and field_info.default is not None:
                    raw_values[name] = field_info.default

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
