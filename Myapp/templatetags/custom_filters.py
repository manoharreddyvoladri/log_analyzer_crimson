from django import template

register = template.Library()

@register.filter
def add_list(value):
    """Add all numbers in a list"""
    try:
        return sum(value)
    except (TypeError, ValueError):
        return 0
