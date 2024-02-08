from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.contrib.contenttypes.models import ContentType
from .models import SwingPoint, DailyPrice, FifteenMinPrice, FiveMinPrice, OneMinPrice

# Signals to delete SwingPoint instances if the associated price instance is deleted.
@receiver(post_delete, sender=DailyPrice)
@receiver(post_delete, sender=FifteenMinPrice)
@receiver(post_delete, sender=FiveMinPrice)
@receiver(post_delete, sender=OneMinPrice)
def delete_swing_point_on_price_delete(sender, instance, **kwargs):
    content_type = ContentType.objects.get_for_model(sender)
    SwingPoint.objects.filter(content_type=content_type, object_id=instance.pk).delete()
