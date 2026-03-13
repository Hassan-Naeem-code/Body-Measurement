"""add ground truth table

Revision ID: 005
Revises: 004
Create Date: 2026-03-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'ground_truth',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('image_path', sa.String(), nullable=False),
        sa.Column('actual_chest_circumference', sa.Float(), nullable=True),
        sa.Column('actual_waist_circumference', sa.Float(), nullable=True),
        sa.Column('actual_hip_circumference', sa.Float(), nullable=True),
        sa.Column('actual_shoulder_width', sa.Float(), nullable=True),
        sa.Column('actual_inseam', sa.Float(), nullable=True),
        sa.Column('actual_arm_length', sa.Float(), nullable=True),
        sa.Column('actual_height', sa.Float(), nullable=True),
        sa.Column('gender', sa.String(), nullable=True),
        sa.Column('age_group', sa.String(), nullable=True),
        sa.Column('body_type', sa.String(), nullable=True),
        sa.Column('pose_type', sa.String(), nullable=True),
        sa.Column('image_quality', sa.String(), nullable=True),
        sa.Column('measured_by', sa.String(), nullable=True),
        sa.Column('measurement_date', sa.String(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )


def downgrade():
    op.drop_table('ground_truth')
