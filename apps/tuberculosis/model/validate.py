from autumn.tool_kit import schema_builder as sb

validate_params = sb.build_validator(
    # Country info
    iso3=str,
    region=sb.Nullable(str),
    pop_region_override=sb.Nullable(str),
    pop_year=int,
    # Running time.
    start_time=float,
    end_time=float,
    time_step=float,
    # disease model
    contact_rate=float,
    recovery_rate= float,
)
