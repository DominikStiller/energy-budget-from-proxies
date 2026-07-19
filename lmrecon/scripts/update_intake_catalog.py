from __future__ import annotations

from ecgtools import Builder
from ecgtools.parsers.cmip import parse_cmip6_using_directories

from lmrecon.util import get_base_path


def update_intake_catalog():
    cmip_path = str(get_base_path() / "CMIP6")
    print("Updating intake-esm catalog")

    builder = Builder([cmip_path], exclude_patterns=["*/catalog*"], depth=100)
    builder.build(parsing_func=parse_cmip6_using_directories)

    print(f"Found {len(builder.df)} files")

    builder.save(
        name="catalog",
        directory=cmip_path,
        path_column_name="path",
        variable_column_name="variable_id",
        data_format="netcdf",
        groupby_attrs=[
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "table_id",
            "grid_label",
        ],
        aggregations=[
            {"type": "union", "attribute_name": "variable_id"},
            {
                "type": "join_existing",
                "attribute_name": "time_range",
                "options": {"dim": "time", "coords": "minimal", "compat": "override"},
            },
            {
                "type": "join_new",
                "attribute_name": "member_id",
                "options": {"coords": "minimal", "compat": "override"},
            },
            {
                "type": "join_new",
                "attribute_name": "dcpp_init_year",
                "options": {"coords": "minimal", "compat": "override"},
            },
        ],
    )


if __name__ == "__main__":
    update_intake_catalog()
