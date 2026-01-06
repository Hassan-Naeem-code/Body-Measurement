"""
A/B Testing Framework for ML vs Fixed Ratios
Enables controlled testing and metrics tracking
"""

import random
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for A/B test"""
    test_name: str
    ml_ratio_percentage: float  # 0-100, percentage of users getting ML ratios
    enabled: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metadata: Optional[Dict] = None


@dataclass
class ABTestAssignment:
    """User assignment to A/B test group"""
    user_id: Optional[str]
    session_id: Optional[str]
    group: str  # 'ml' or 'fixed'
    assigned_at: datetime
    test_name: str


class ABTestingFramework:
    """
    A/B Testing framework for ML ratio experiments

    Features:
    - Configurable traffic split (e.g., 50% ML, 50% Fixed)
    - Consistent assignment (same user always gets same variant)
    - Easy tracking and analytics
    - Gradual rollout support
    """

    def __init__(self, config: ABTestConfig):
        """
        Initialize A/B testing framework

        Args:
            config: A/B test configuration
        """
        self.config = config
        self.assignments = {}  # user_id -> assignment

    def assign_variant(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ABTestAssignment:
        """
        Assign user to ML or Fixed variant

        Args:
            user_id: Unique user identifier (for consistent assignment)
            session_id: Session identifier (fallback if no user_id)

        Returns:
            ABTestAssignment with group assignment
        """
        # Check if test is enabled
        if not self.config.enabled:
            return ABTestAssignment(
                user_id=user_id,
                session_id=session_id,
                group='fixed',  # Default to fixed if test disabled
                assigned_at=datetime.utcnow(),
                test_name=self.config.test_name
            )

        # Check date range
        now = datetime.utcnow()
        if self.config.start_date and now < self.config.start_date:
            return ABTestAssignment(
                user_id=user_id,
                session_id=session_id,
                group='fixed',  # Before test starts
                assigned_at=now,
                test_name=self.config.test_name
            )

        if self.config.end_date and now > self.config.end_date:
            return ABTestAssignment(
                user_id=user_id,
                session_id=session_id,
                group='ml',  # After test ends, use ML (winner)
                assigned_at=now,
                test_name=self.config.test_name
            )

        # Get identifier for consistent hashing
        identifier = user_id or session_id or str(random.random())

        # Check if already assigned
        if identifier in self.assignments:
            return self.assignments[identifier]

        # Assign to variant using consistent hashing
        group = self._assign_group(identifier)

        assignment = ABTestAssignment(
            user_id=user_id,
            session_id=session_id,
            group=group,
            assigned_at=now,
            test_name=self.config.test_name
        )

        # Cache assignment
        self.assignments[identifier] = assignment

        logger.info(
            f"AB Test Assignment - Test: {self.config.test_name}, "
            f"User: {identifier[:8]}..., Group: {group}"
        )

        return assignment

    def _assign_group(self, identifier: str) -> str:
        """
        Assign group using consistent hashing

        Args:
            identifier: User or session identifier

        Returns:
            'ml' or 'fixed'
        """
        # Use hash of identifier for consistent assignment
        hash_value = hash(identifier) % 100

        # Assign based on percentage
        if hash_value < self.config.ml_ratio_percentage:
            return 'ml'
        else:
            return 'fixed'

    def should_use_ml(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Simplified method to check if ML should be used

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            True if ML variant, False if Fixed variant
        """
        assignment = self.assign_variant(user_id, session_id)
        return assignment.group == 'ml'

    def get_variant_stats(self) -> Dict:
        """Get statistics about variant assignments"""
        if not self.assignments:
            return {
                'total_assignments': 0,
                'ml_count': 0,
                'fixed_count': 0,
                'ml_percentage': 0.0,
            }

        ml_count = sum(1 for a in self.assignments.values() if a.group == 'ml')
        fixed_count = len(self.assignments) - ml_count

        return {
            'total_assignments': len(self.assignments),
            'ml_count': ml_count,
            'fixed_count': fixed_count,
            'ml_percentage': (ml_count / len(self.assignments) * 100),
            'test_name': self.config.test_name,
            'config_ml_percentage': self.config.ml_ratio_percentage,
        }


# Predefined test configurations

class ABTestPresets:
    """Preset A/B test configurations for common scenarios"""

    @staticmethod
    def fifty_fifty() -> ABTestConfig:
        """50/50 split between ML and Fixed"""
        return ABTestConfig(
            test_name="ml_vs_fixed_50_50",
            ml_ratio_percentage=50.0,
            enabled=True,
            metadata={
                'description': '50/50 split test',
                'goal': 'Compare ML vs Fixed with equal traffic',
            }
        )

    @staticmethod
    def gradual_rollout_20() -> ABTestConfig:
        """20% ML, 80% Fixed (conservative rollout)"""
        return ABTestConfig(
            test_name="ml_gradual_rollout_20",
            ml_ratio_percentage=20.0,
            enabled=True,
            metadata={
                'description': 'Conservative ML rollout',
                'goal': 'Test ML with 20% traffic before full rollout',
            }
        )

    @staticmethod
    def gradual_rollout_50() -> ABTestConfig:
        """50% ML, 50% Fixed"""
        return ABTestConfig(
            test_name="ml_gradual_rollout_50",
            ml_ratio_percentage=50.0,
            enabled=True,
            metadata={
                'description': 'Half traffic ML rollout',
                'goal': 'Test ML with 50% traffic',
            }
        )

    @staticmethod
    def gradual_rollout_80() -> ABTestConfig:
        """80% ML, 20% Fixed (aggressive rollout)"""
        return ABTestConfig(
            test_name="ml_gradual_rollout_80",
            ml_ratio_percentage=80.0,
            enabled=True,
            metadata={
                'description': 'Aggressive ML rollout',
                'goal': 'Test ML with 80% traffic before full rollout',
            }
        )

    @staticmethod
    def ml_only() -> ABTestConfig:
        """100% ML (full rollout)"""
        return ABTestConfig(
            test_name="ml_full_rollout",
            ml_ratio_percentage=100.0,
            enabled=True,
            metadata={
                'description': 'Full ML rollout',
                'goal': '100% of traffic uses ML ratios',
            }
        )

    @staticmethod
    def fixed_only() -> ABTestConfig:
        """100% Fixed (baseline/rollback)"""
        return ABTestConfig(
            test_name="fixed_baseline",
            ml_ratio_percentage=0.0,
            enabled=True,
            metadata={
                'description': 'Baseline/rollback mode',
                'goal': '100% of traffic uses fixed ratios',
            }
        )


# Global AB test instance (can be configured via environment or settings)
_global_ab_test: Optional[ABTestingFramework] = None


def get_ab_test() -> ABTestingFramework:
    """Get global A/B test instance"""
    global _global_ab_test

    if _global_ab_test is None:
        # Default: 50/50 split
        config = ABTestPresets.fifty_fifty()
        _global_ab_test = ABTestingFramework(config)

    return _global_ab_test


def configure_ab_test(config: ABTestConfig):
    """Configure global A/B test"""
    global _global_ab_test
    _global_ab_test = ABTestingFramework(config)
    logger.info(f"Configured A/B test: {config.test_name} ({config.ml_ratio_percentage}% ML)")


def should_use_ml_ratios(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> bool:
    """
    Simplified global function to check if ML should be used

    Args:
        user_id: User identifier
        session_id: Session identifier

    Returns:
        True if ML variant, False if Fixed variant
    """
    ab_test = get_ab_test()
    return ab_test.should_use_ml(user_id, session_id)
