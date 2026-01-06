"""add ml tracking to measurements

Revision ID: 004
Revises: 003
Create Date: 2026-01-06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade():
    # Add used_ml_ratios column to measurements table
    op.add_column('measurements', sa.Column('used_ml_ratios', postgresql.JSON(astext_type=sa.Text()), nullable=True))


def downgrade():
    # Remove used_ml_ratios column
    op.drop_column('measurements', 'used_ml_ratios')
