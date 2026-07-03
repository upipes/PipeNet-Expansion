def decimal_to_float(value):
    return None if value is None else float(value)


def number_label(min_value, max_value, unit="", precision=None):
    if min_value is None and max_value is None:
        return ""
    if min_value is None:
        return f"{max_value:g}{unit}"
    if max_value is None:
        return f"{min_value:g}{unit}"
    if min_value == max_value:
        return f"{min_value:g}{unit}"
    return f"{min_value:g}-{max_value:g}{unit}"


def supported_classes_for(domain):
    stored_classes = list(domain.semantic_categories.order_by("id").values_list("name", flat=True))
    if stored_classes:
        return stored_classes
    if domain.domain_type == "soil":
        return ["Cavity", "Crack", "Metal Pipeline"]
    return ["Cavity", "Crack", "Loose", "Normal", "Pipeline"]


def serialize_domain_summary(domain):
    payload = {
        "id": domain.id,
        "code": domain.code,
        "name": domain.name,
        "domainType": domain.domain_type,
        "renderMode": domain.domain_type,
        "supportedClasses": supported_classes_for(domain),
        "displayOrder": domain.display_order,
    }

    if domain.domain_type == "road":
        payload["preview"] = {
            "condition": domain.condition_text,
            "roadSurface": domain.road_surface,
            "frequencyRange": {
                "min": domain.frequency_min,
                "max": domain.frequency_max,
                "unit": "MHz",
                "label": number_label(domain.frequency_min, domain.frequency_max, " MHz"),
            },
            "timeWindow": {
                "value": domain.time_window_ns,
                "unit": "ns",
                "label": "" if domain.time_window_ns is None else f"{domain.time_window_ns} ns",
            },
        }
    else:
        payload["preview"] = {
            "composition": {
                "sand": decimal_to_float(domain.sand_percent),
                "silt": decimal_to_float(domain.silt_percent),
                "clay": decimal_to_float(domain.clay_percent),
            },
            "waterContent": {
                "min": decimal_to_float(domain.water_min),
                "max": decimal_to_float(domain.water_max),
                "unit": "%",
                "label": number_label(domain.water_min, domain.water_max, "%"),
            },
            "relativePermittivity": {
                "min": decimal_to_float(domain.permittivity_min),
                "max": decimal_to_float(domain.permittivity_max),
                "label": number_label(domain.permittivity_min, domain.permittivity_max),
            },
            "conductivity": {
                "min": decimal_to_float(domain.conductivity_min),
                "max": decimal_to_float(domain.conductivity_max),
                "unit": "S/m",
                "label": number_label(domain.conductivity_min, domain.conductivity_max, " S/m"),
            },
            "peplinskiDimension": decimal_to_float(domain.peplinski_dimension),
        }

    return payload


def serialize_domain_detail(domain):
    payload = serialize_domain_summary(domain)
    payload["detail"] = {
        "areaDescription": domain.area_description,
        "signalBehavior": domain.signal_behavior,
        "semanticUsage": domain.semantic_usage,
        "createdAt": domain.created_at.isoformat() if domain.created_at else None,
        "updatedAt": domain.updated_at.isoformat() if domain.updated_at else None,
    }
    return payload

