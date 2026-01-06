"""Add products and size charts tables

Revision ID: 003
Revises: 002
Create Date: 2025-01-05

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    # Create products table
    op.create_table(
        'products',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('brand_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('brands.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('sku', sa.String(100), nullable=True),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('subcategory', sa.String(100), nullable=True),
        sa.Column('gender', sa.String(20), nullable=True),  # 'male', 'female', 'unisex'
        sa.Column('age_group', sa.String(20), nullable=True),  # 'adult', 'teen', 'child'
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('image_url', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
    )

    # Create size_charts table
    op.create_table(
        'size_charts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('product_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('products.id', ondelete='CASCADE'), nullable=False),
        sa.Column('size_name', sa.String(20), nullable=False),  # 'XS', 'S', 'M', 'L', etc.
        sa.Column('chest_min', sa.Float, nullable=True),
        sa.Column('chest_max', sa.Float, nullable=True),
        sa.Column('waist_min', sa.Float, nullable=True),
        sa.Column('waist_max', sa.Float, nullable=True),
        sa.Column('hip_min', sa.Float, nullable=True),
        sa.Column('hip_max', sa.Float, nullable=True),
        sa.Column('height_min', sa.Float, nullable=True),
        sa.Column('height_max', sa.Float, nullable=True),
        sa.Column('inseam_min', sa.Float, nullable=True),
        sa.Column('inseam_max', sa.Float, nullable=True),
        sa.Column('weight_min', sa.Float, nullable=True),
        sa.Column('weight_max', sa.Float, nullable=True),
        sa.Column('fit_type', sa.String(20), default='regular'),  # 'tight', 'regular', 'loose'
        sa.Column('display_order', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('product_id', 'size_name', name='unique_product_size'),
    )

    # Create indexes for faster lookups
    op.create_index('idx_products_brand_id', 'products', ['brand_id'])
    op.create_index('idx_products_category', 'products', ['category'])
    op.create_index('idx_products_active', 'products', ['is_active'])
    op.create_index('idx_size_charts_product_id', 'size_charts', ['product_id'])

    # Add product_id to measurements table (optional - for tracking which product was measured for)
    op.add_column('measurements', sa.Column('product_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('products.id', ondelete='SET NULL'), nullable=True))
    op.create_index('idx_measurements_product_id', 'measurements', ['product_id'])


def downgrade():
    op.drop_index('idx_measurements_product_id', 'measurements')
    op.drop_column('measurements', 'product_id')

    op.drop_index('idx_size_charts_product_id', 'size_charts')
    op.drop_index('idx_products_active', 'products')
    op.drop_index('idx_products_category', 'products')
    op.drop_index('idx_products_brand_id', 'products')

    op.drop_table('size_charts')
    op.drop_table('products')
