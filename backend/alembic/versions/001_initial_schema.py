"""Initial schema - brands table

Revision ID: 001
Revises:
Create Date: 2025-01-05

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create brands table
    op.create_table(
        'brands',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('api_key', sa.String(64), nullable=False, unique=True),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
    )

    # Create indexes
    op.create_index('idx_brands_api_key', 'brands', ['api_key'])
    op.create_index('idx_brands_email', 'brands', ['email'])


def downgrade():
    op.drop_index('idx_brands_email', 'brands')
    op.drop_index('idx_brands_api_key', 'brands')
    op.drop_table('brands')
