"""Agent factory related classes."""

import core

from .base_agent import LlmEnabledAgent
from .claim_agent import ClaimAgent
from .claim_appeal_agent import ClaimAppealAgent
from .claim_explanation_agent import ClaimExplanationAgent
from .customer_agent import CustomerAgent
from .policy_rule_agent import PolicyRuleAgent


class AgentFactory(core.ConfigurableObjectFactory):
    """Factory class for creating and caching agent objects."""

    _TYPES_MAPPING = {
        "claim": ClaimAgent,
        "claim_appeal": ClaimAppealAgent,
        "claim_explanation": ClaimExplanationAgent,
        "customer": CustomerAgent,
        "policy_rule": PolicyRuleAgent,
    }

    async def _create_obj_async(self, _id: str, config: dict) -> core.Configurable:
        """Create agent instance asynchronously for LLM-backed agents.

        Args:
            _id (str): Identifier.
            config (dict): Provided configuration container.

        Returns:
            core.Configurable: Concrete agent instance.
        """
        if _id not in self._TYPES_MAPPING:
            raise ValueError(f"Provided identifier is not supported: {_id}")
        configurable_type = self._TYPES_MAPPING[_id]
        if issubclass(configurable_type, LlmEnabledAgent):
            return await configurable_type.create(
                configurable_type.config_data_type()(**config))
        return self._create_obj(_id, config)
