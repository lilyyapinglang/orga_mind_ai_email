"""Persist provenance for AI-generated drafts."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("core", "0035_address_normalization")]

    operations = [
        migrations.AddField(
            model_name="message",
            name="ai_metadata",
            field=models.JSONField(
                blank=True,
                help_text="Grounding sources and review flags for an AI-generated draft.",
                null=True,
                verbose_name="AI metadata",
            ),
        )
    ]
