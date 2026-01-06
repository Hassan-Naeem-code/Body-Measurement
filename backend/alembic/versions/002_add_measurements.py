"""Add measurements table

Revision ID: 002
Revises: 001
Create Date: 2025-01-05

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Create measurements table
    op.create_table(
        'measurements',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('brand_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('brands.id'), nullable=False),

        # Body measurements in cm
        sa.Column('shoulder_width', sa.Float, nullable=False),
        sa.Column('chest_width', sa.Float, nullable=False),
        sa.Column('waist_width', sa.Float, nullable=False),
        sa.Column('hip_width', sa.Float, nullable=False),
        sa.Column('inseam', sa.Float, nullable=False),
        sa.Column('arm_length', sa.Float, nullable=False),

        # Confidence scores (JSON)
        sa.Column('confidence_scores', postgresql.JSON, nullable=False),

        # Size recommendation
        sa.Column('recommended_size', sa.String, nullable=True),
        sa.Column('size_probabilities', postgresql.JSON, nullable=True),

        # Processing metadata
        sa.Column('processing_time_ms', sa.Float, nullable=False),
        sa.Column('image_hash', sa.String, nullable=True),

        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
    )

    # Create indexes
    op.create_index('idx_measurements_brand_id', 'measurements', ['brand_id'])
    op.create_index('idx_measurements_created_at', 'measurements', ['created_at'])


def downgrade():
    op.drop_index('idx_measurements_created_at', 'measurements')
    op.drop_index('idx_measurements_brand_id', 'measurements')
    op.drop_table('measurements')
