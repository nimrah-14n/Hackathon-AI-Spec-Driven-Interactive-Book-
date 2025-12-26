from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class FeatureFlagsService:
    """
    Service to manage feature flags for optional components
    """

    @staticmethod
    def is_enabled(feature_name: str) -> bool:
        """
        Check if a feature is enabled based on settings
        """
        try:
            if feature_name == "selected_text_mode":
                return settings.selected_text_mode_enabled
            elif feature_name == "auth" or feature_name == "authentication":
                return settings.auth_enabled
            elif feature_name == "personalization":
                return settings.personalization_enabled
            elif feature_name == "urdu_translation":
                return settings.urdu_translation_enabled
            else:
                logger.warning(f"Unknown feature flag requested: {feature_name}")
                return False
        except AttributeError:
            logger.error(f"Feature flag attribute not found for: {feature_name}")
            return False

    @staticmethod
    def get_enabled_features() -> dict:
        """
        Get dictionary of all feature flags and their status
        """
        return {
            "selected_text_mode": settings.selected_text_mode_enabled,
            "auth": settings.auth_enabled,
            "personalization": settings.personalization_enabled,
            "urdu_translation": settings.urdu_translation_enabled
        }

    @staticmethod
    def check_feature_access(feature_name: str) -> bool:
        """
        Check if a feature is accessible (enabled and properly configured)
        """
        if not FeatureFlagsService.is_enabled(feature_name):
            return False

        # Additional checks for specific features
        if feature_name == "auth" and not settings.openrouter_api_key:
            logger.warning("Auth feature enabled but OpenRouter API key not configured")
            return False
        elif feature_name == "urdu_translation" and not settings.openrouter_api_key:
            logger.warning("Urdu translation feature enabled but OpenRouter API key not configured")
            return False

        return True


# Global instance for easy access
feature_flags = FeatureFlagsService()