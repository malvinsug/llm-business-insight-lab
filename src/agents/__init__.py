"""Agent Module: Dual-agent architecture for business insight generation.

Implements:
1. LogicAnalystAgent: Symbolic reasoning with First-Order Logic (SymCoT)
2. StrategicConsultantAgent: Direct strategic synthesis (NoCoT)
3. DualAgentOrchestrator: Orchestrates agent coordination and result aggregation
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import json
import time
from dataclasses import dataclass

try:
    from pydantic_ai import Agent, RunContext
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False


from loguru import logger
from src.data_models import PromptStrategy, ExperimentConfig, EvaluationMetrics


@dataclass
class AgentResponse:
    """Response structure from an agent execution."""
    strategy: PromptStrategy
    content: str
    reasoning: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {"input": 0, "output": 0}


class BaseAgent(ABC):
    """Abstract base class for LLM agents in the framework."""
    
    def __init__(
        self,
        model_name: str,
        strategy: PromptStrategy,
        api_key: str,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            model_name: LLM model identifier
            strategy: Prompting strategy (NoCoT, CoT, SymCoT)
            api_key: API key for LLM provider
            system_prompt: Optional custom system prompt
        """
        self.model_name = model_name
        self.strategy = strategy
        self.api_key = api_key
        self.system_prompt = system_prompt or self._default_system_prompt()
        logger.debug(f"Initializing {self.__class__.__name__} with strategy={strategy.value}")
    
    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return default system prompt for this agent."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        user_question: str,
        context: Dict[str, Any],
        config: ExperimentConfig,
    ) -> AgentResponse:
        """
        Generate response to business question.
        
        Args:
            user_question: KPI question from user
            context: Contextual data (KPIs, metrics, etc.)
            config: Experiment configuration
        
        Returns:
            AgentResponse with content, reasoning, and latency
        """
        pass


class LogicAnalystAgent(BaseAgent):
    """
    Logic Analyst Agent: Symbolic Chain of Thought (SymCoT) with First-Order Logic.
    
    This agent specializes in rigorous, fact-based analysis by:
    1. Extracting discrete facts and observations from data
    2. Formalizing them as First-Order Logic predicates
    3. Verifying logical consistency before generating recommendations
    
    Mathematical Approach:
    - Observations are formalized as predicates: e.g., P(kpi_x, quarter_y, trend_z)
    - Logical implications: if P(trend_increasing) ∧ threshold_exceeded(x) → action_required()
    - Verification step: confirms all statements can be grounded in data before output
    
    This approach reduces hallucinations (high factuality) at the cost of lower flexibility.
    Recommended for: Engineering, Root Cause Analysis, Performance Review
    """
    
    def _default_system_prompt(self) -> str:
        return """\
You are the Logic Analyst Agent—an expert in rigorous, fact-based business analysis.

Your role:
1. Extract ONLY observations backed by data
2. Formalize each observation as a logical fact (predicate)
3. Derive conclusions via logical inference
4. Verify all statements before recommending action

Format your response as:
---OBSERVATIONS---
[List facts backed by KPI data]

---LOGICAL FORMALIZATION---
[Express as First-Order Logic predicates]

---VERIFICATION---
[Confirm each fact is grounded in data]

---RECOMMENDATIONS---
[Actions derived from verified logic]

Prioritize factual accuracy over brevity. When uncertain, flag the uncertainty.
"""
    
    async def generate_response(
        self,
        user_question: str,
        context: Dict[str, Any],
        config: ExperimentConfig,
    ) -> AgentResponse:
        """
        Generate SymCoT response with logical verification.
        
        Args:
            user_question: Input KPI question
            context: Contextual data dict
            config: Experiment config
        
        Returns:
            AgentResponse with reasoning tree
        """
        start_time = time.time()
        
        # Format context for prompt
        context_str = self._format_context(context)
        prompt = f"""\
Question: {user_question}

Business Context:
{context_str}

Provide a rigorous, logic-based analysis following the format specified in your instructions.
Temperature: {config.temperature}
"""
        
        logger.info(f"LogicAnalystAgent: Generating SymCoT response for question: {user_question[:50]}...")
        
        # Simulated LLM call (replace with actual API in production)
        # In production, use pydantic_ai.Agent for structured output
        response_content = await self._call_llm(prompt, config)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract reasoning from response
        reasoning = self._extract_reasoning(response_content)
        
        return AgentResponse(
            strategy=PromptStrategy.SYMCOT,
            content=response_content,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )
    
    async def _call_llm(self, prompt: str, config: ExperimentConfig) -> str:
        """Mock LLM call. Replace with actual API integration."""
        # Placeholder: in production, use openai.AsyncOpenAI or anthropic.AsyncAnthropic
        await __import__("asyncio").sleep(0.1)  # Simulate latency
        return "Logic-verified analysis: [placeholder response]"
    
    def _extract_reasoning(self, response: str) -> Dict[str, Any]:
        """Extract structured reasoning from response."""
        try:
            # Parse response sections
            sections = {
                "observations": self._extract_section(response, "OBSERVATIONS"),
                "formalization": self._extract_section(response, "LOGICAL FORMALIZATION"),
                "verification": self._extract_section(response, "VERIFICATION"),
                "recommendations": self._extract_section(response, "RECOMMENDATIONS"),
            }
            return sections
        except Exception as e:
            logger.warning(f"Failed to extract reasoning: {e}")
            return {}
    
    @staticmethod
    def _extract_section(response: str, section_name: str) -> str:
        """Extract a named section from response (e.g., '---OBSERVATIONS---')."""
        try:
            start = response.find(f"---{section_name}---")
            if start == -1:
                return ""
            start += len(f"---{section_name}---")
            end = response.find("---", start)
            if end == -1:
                end = len(response)
            return response[start:end].strip()
        except Exception:
            return ""
    
    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        """Format context dictionary for inclusion in prompt."""
        lines = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                lines.append(f"{key}:\n{json.dumps(value, indent=2)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


class StrategicConsultantAgent(BaseAgent):
    """
    Strategic Consultant Agent: Direct synthesis without structured reasoning (NoCoT).
    
    This agent generates high-level strategic recommendations by:
    1. Synthesizing key insights directly without intermediate steps
    2. Prioritizing actionability and strategic alignment over rigor
    3. Leveraging LLM's world models for forward-looking analysis
    
    Key Insight from Thesis:
    "Contrary to common belief, complex reasoning chains like CoT can underperform
    compared to a direct NoCoT baseline in high-level strategic tasks. This is the
    'Guidance Bias' paradox: over-structuring prompts can constrain the model's
    broader pattern recognition, reducing performance in strategic synthesis."
    
    Recommended for: Marketing, Strategic Planning, Executive Briefings
    """
    
    def _default_system_prompt(self) -> str:
        return """\
You are the Strategic Consultant Agent—an expert in forward-looking business strategy.

Your role:
1. Synthesize key insights directly from KPI data
2. Identify strategic opportunities and risks
3. Recommend actions that drive business value
4. Prioritize clarity and actionability

Provide concise, high-level recommendations without verbose reasoning chains.
Focus on "what to do" and "why it matters for the business."

Format:
- Key Insights: [2-3 main takeaways]
- Strategic Opportunities: [Actionable recommendations]
- Risk Mitigation: [Potential challenges and responses]
- Next Steps: [Immediate actions]
"""
    
    async def generate_response(
        self,
        user_question: str,
        context: Dict[str, Any],
        config: ExperimentConfig,
    ) -> AgentResponse:
        """
        Generate NoCoT response with direct synthesis.
        
        Args:
            user_question: Input KPI question
            context: Contextual data dict
            config: Experiment config
        
        Returns:
            AgentResponse with strategic recommendations
        """
        start_time = time.time()
        
        context_str = self._format_context(context)
        prompt = f"""\
Strategic Question: {user_question}

Business Context:
{context_str}

Provide strategic recommendations following the format in your instructions.
Temperature: {config.temperature}
"""
        
        logger.info(f"StrategicConsultantAgent: Generating NoCoT response for: {user_question[:50]}...")
        
        response_content = await self._call_llm(prompt, config)
        latency_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            strategy=PromptStrategy.NOCOT,
            content=response_content,
            reasoning=None,  # NoCoT doesn't extract structured reasoning
            latency_ms=latency_ms,
        )
    
    async def _call_llm(self, prompt: str, config: ExperimentConfig) -> str:
        """Mock LLM call. Replace with actual API integration."""
        await __import__("asyncio").sleep(0.1)  # Simulate latency
        return "Strategic recommendations: [placeholder response]"
    
    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        lines = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                lines.append(f"{key}:\n{json.dumps(value, indent=2)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


class DualAgentOrchestrator:
    """
    Orchestrates dual-agent system for comprehensive business analysis.
    
    Workflow:
    1. Route question to appropriate agent based on context
    2. Execute both agents in parallel for comparison
    3. Aggregate results with metrics
    4. Return structured ExperimentResult
    """
    
    def __init__(
        self,
        logic_analyst: LogicAnalystAgent,
        strategic_consultant: StrategicConsultantAgent,
    ):
        self.logic_analyst = logic_analyst
        self.strategic_consultant = strategic_consultant
        logger.info("DualAgentOrchestrator initialized with Logic Analyst and Strategic Consultant")
    
    async def orchestrate(
        self,
        user_question: str,
        context: Dict[str, Any],
        config: ExperimentConfig,
        target_strategies: Optional[list] = None,
    ) -> Dict[PromptStrategy, AgentResponse]:
        """
        Execute agents based on target strategies.
        
        Args:
            user_question: Business question
            context: Contextual data
            config: Experiment config
            target_strategies: List of strategies to execute (default: both)
        
        Returns:
            Dictionary mapping strategies to responses
        """
        if target_strategies is None:
            target_strategies = [PromptStrategy.SYMCOT, PromptStrategy.NOCOT]
        
        results = {}
        
        import asyncio
        
        tasks = []
        if PromptStrategy.SYMCOT in target_strategies:
            tasks.append(
                self.logic_analyst.generate_response(user_question, context, config)
            )
        if PromptStrategy.NOCOT in target_strategies:
            tasks.append(
                self.strategic_consultant.generate_response(user_question, context, config)
            )
        
        # Execute in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Agent execution failed: {response}")
            else:
                results[response.strategy] = response
        
        return results
