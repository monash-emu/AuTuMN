Feedback Type:
Frown (Error)

Timestamp:
2021-09-06T03:13:41.0103673Z

Local Time:
2021-09-06T13:13:41.0103673+10:00

Session ID:
9c309f94-a13e-46e4-a87d-d69548bf638a

Release:
August 2021

Product Version:
2.96.1061.0 (21.08) (x64)

OS Version:
Microsoft Windows NT 10.0.17763.0 (x64 en-GB)

CLR Version:
4.7 or later [Release Number = 528049]

Peak Virtual Memory:
103 GB

Private Memory:
891 MB

Peak Working Set:
1.1 GB

IE Version:
11.1790.17763.0

User ID:
bba3e10e-bf65-455b-b5ce-ad1c3b84a97c

Workbook Package Info:
1* - en-AU, Query Groups: 7, fastCombine: Disabled, runBackgroundAnalysis: False.

Telemetry Enabled:
False

Snapshot Trace Logs:
C:\Users\maba0001\Microsoft\Power BI Desktop Store App\FrownSnapShot8d4e69b4-9ac1-47a0-9d75-078a58e4a135.zip

Model Default Mode:
Import

Model Version:
PowerBI_V3

Performance Trace Logs:
C:\Users\maba0001\Microsoft\Power BI Desktop Store App\PerformanceTraces.zip

Enabled Preview Features:
PBI_shapeMapVisualEnabled
PBI_JsonTableInference
PBI_NewWebTableInference
PBI_ImportTextByExample
PBI_ExcelTableInference
PBI_azureMapVisual
PBI_compositeModelsOverAS
PBI_dynamicParameters
PBI_rdlNativeVisual

Disabled Preview Features:
PBI_SpanishLinguisticsEnabled
PBI_qnaLiveConnect
PBI_dataPointLassoSelect
PBI_enhancedTooltips

Disabled DirectQuery Options:
TreatHanaAsRelationalSource

Cloud:
GlobalCloud

DPI Scale:
125%

Supported Services:
Power BI

Formulas:

section Section1;

shared base_build_index =
    let
        Source = Folder.Files(data_location),
        #"Removed Other Columns" =
            Table.SelectColumns(
                Source,
                {
                    "Folder Path",
                    "Name"
                }
            ),
        #"Merged Columns" =
            Table.CombineColumns(
                #"Removed Other Columns",
                {
                    "Folder Path",
                    "Name"
                },
                Combiner.CombineTextByDelimiter(
                    "",
                    QuoteStyle.None
                ),
                "Full_Path"
            ),
        #"Invoked Custom Function" =
            Table.AddColumn(
                #"Merged Columns",
                "build_info",
                each GetBuild([Full_Path])
            ),
        #"Expanded build_info1" =
            Table.ExpandTableColumn(
                #"Invoked Custom Function",
                "build_info",
                {
                    "unix_epoch_time",
                    "git_commit",
                    "app_name",
                    "region_name",
                    "build_time",
                    "build_date"
                },
                {
                    "unix_epoch_time",
                    "git_commit",
                    "app_name",
                    "region_name",
                    "build_time",
                    "build_date"
                }
            ),
        #"Changed Type1" =
            Table.TransformColumnTypes(
                #"Expanded build_info1",
                {
                    {
                        "build_time",
                        type datetime
                    },
                    {
                        "unix_epoch_time",
                        Int64.Type
                    },
                    {
                        "git_commit",
                        type text
                    },
                    {
                        "app_name",
                        type text
                    },
                    {
                        "region_name",
                        type text
                    },
                    {
                        "build_date",
                        type date
                    }
                }
            ),
        #"Sorted Rows" =
            Table.Sort(
                #"Changed Type1",
                {
                    {
                        "unix_epoch_time",
                        Order.Ascending
                    }
                }
            ),
        #"Added Index" =
            Table.AddIndexColumn(
                #"Sorted Rows",
                "build_index",
                1,
                1,
                Int64.Type
            ),
        Custom1 =
            Table.AddKey(
                #"Added Index",
                {"build_index"},
                true
            ),
        #"Replaced Value" =
            Table.ReplaceValue(
                Custom1,
                "-",
                " ",
                Replacer.ReplaceText,
                {"region_name"}
            ),
        #"Replaced Value1" =
            Table.ReplaceValue(
                #"Replaced Value",
                "manila",
                "national capital region",
                Replacer.ReplaceText,
                {"region_name"}
            )
    in
        #"Replaced Value1";

shared base_powerbi_output =
    let
        Source = base_build_index,
        #"Invoked Custom Function" =
            Table.AddColumn(
                Source,
                "GetPowerbiOutputs",
                each GetPowerbiOutputs([Full_Path])
            ),
        #"Removed Columns" =
            Table.RemoveColumns(
                #"Invoked Custom Function",
                {
                    "app_name",
                    "Full_Path",
                    "git_commit",
                    "region_name",
                    "unix_epoch_time",
                    "build_time",
                    "build_date"
                }
            ),
        #"Removed Errors" =
            Table.RemoveRowsWithErrors(
                #"Removed Columns",
                {"GetPowerbiOutputs"}
            ),
        #"Expanded GetPowerbiOutputs" =
            Table.ExpandTableColumn(
                #"Removed Errors",
                "GetPowerbiOutputs",
                {
                    "pbi_scenario",
                    "pbi_date",
                    "pbi_value",
                    "dim_age",
                    "pbi_clinical",
                    "pbi_compartment",
                    "immunity"
                },
                {
                    "pbi_scenario",
                    "pbi_date",
                    "pbi_value",
                    "dim_age",
                    "pbi_clinical",
                    "pbi_compartment",
                    "immunity"
                }
            )
            as table,
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Expanded GetPowerbiOutputs",
                {
                    {
                        "dim_age",
                        Int64.Type
                    },
                    {
                        "pbi_scenario",
                        type text
                    },
                    {
                        "pbi_compartment",
                        type text
                    },
                    {
                        "pbi_clinical",
                        type text
                    },
                    {
                        "pbi_date",
                        type date
                    },
                    {
                        "pbi_value",
                        type number
                    },
                    {
                        "immunity",
                        type text
                    }
                }
            ),
        #"Inserted Merged Column" =
            Table.AddColumn(
                #"Changed Type",
                "bld_scn",
                each
                    Text.Combine(
                        {
                            Text.From([build_index], "en-AU"),
                            [pbi_scenario]
                        },
                        "_"
                    ),
                type text
            )
    in
        #"Inserted Merged Column";

shared #"#shared functions" =
    let
        Source = #shared,
        #"Converted to Table" = Record.ToTable(Source),
        #"Filtered Rows" =
            Table.SelectRows(
                #"Converted to Table",
                each ([Name] = "Table.ExpandTableColumn")
            )
    in
        #"Filtered Rows";

shared Query_Diagnostics =
    (Input as table) as table =>
        let
            Source = Input,
            #"Expanded Additional Info" =
                Table.ExpandRecordColumn(
                    Source,
                    "Additional Info",
                    {"Message"},
                    {"Message"}
                ),
            #"Calculated Total Seconds" =
                Table.TransformColumns(
                    #"Expanded Additional Info",
                    {
                        {
                            "Exclusive Duration",
                            Duration.TotalSeconds,
                            type number
                        }
                    }
                ),
            #"Sorted Rows" =
                Table.Sort(
                    #"Calculated Total Seconds",
                    {
                        {
                            "Id",
                            Order.Ascending
                        },
                        {
                            "Start Time",
                            Order.Ascending
                        }
                    }
                ),
            #"Removed Other Columns" =
                Table.SelectColumns(
                    #"Sorted Rows",
                    {
                        "Id",
                        "Query",
                        "Category",
                        "Operation",
                        "Start Time",
                        "End Time",
                        "Exclusive Duration (%)",
                        "Exclusive Duration",
                        "Data Source Query",
                        "Message",
                        "Row Count",
                        "Content Length",
                        "Path",
                        "Group Id"
                    }
                ),
            #"Changed Type" =
                Table.TransformColumnTypes(
                    #"Removed Other Columns",
                    {
                        {
                            "Message",
                            type text
                        }
                    }
                ),
            #"Replaced Value" =
                Table.ReplaceValue(
                    #"Changed Type",
                    null,
                    "Missing",
                    Replacer.ReplaceValue,
                    {"Path"}
                ),
            BufferedTable = Table.Buffer(#"Replaced Value"),
            GetAllChildRows =
                (CurrentId, CurrentPath) =>
                    Table.SelectRows(
                        BufferedTable,
                        each
                            [Path]
                            <> "Missing"
                            and [Id]
                            = CurrentId
                            and Text.StartsWith([Path], CurrentPath)
                    ),
            AddTotalED =
                Table.AddColumn(
                    #"Replaced Value",
                    "Exclusive Duration (Including Child Operations)",
                    each List.Sum(GetAllChildRows([Id], [Path]    )[Exclusive Duration]),
                    type number
                ),
            AddTotalEDPct =
                Table.AddColumn(
                    AddTotalED,
                    "Exclusive Duration (%) (Including Child Operations)",
                    each List.Sum(GetAllChildRows([Id], [Path])[#"Exclusive Duration (%)"]),
                    Percentage.Type
                ),
            #"Inserted Text Before Delimiter" =
                Table.AddColumn(
                    AddTotalEDPct,
                    "Parent Path",
                    each
                        Text.BeforeDelimiter(
                            [Path],
                            "/",
                            {
                                0,
                                RelativePosition.FromEnd
                            }
                        ),
                    type text
                ),
            #"Added Custom" =
                Table.AddColumn(
                    #"Inserted Text Before Delimiter",
                    "Child Operations",
                    each
                        let
                            CurrentPath = [Path],
                            CurrentId = [Id],
                            ChildRows =
                                Table.SelectRows(
                                    @#"Added Custom",
                                    each
                                        [Path]
                                        <> "Missing"
                                        and [Parent Path]
                                        = CurrentPath
                                        and [Id]
                                        = CurrentId
                                ),
                            Output =
                                if Table.RowCount(ChildRows) = 0 then
                                    null
                                else
                                    ChildRows
                        in
                            Output
                ),
            #"Filtered Rows" =
                Table.SelectRows(
                    #"Added Custom",
                    each ([Path] = "0" or [Path] = "Missing")
                ),
            #"Removed Columns" =
                Table.RemoveColumns(
                    #"Filtered Rows",
                    {"Parent Path"}
                )
        in
            #"Removed Columns";

shared #"last refresh" =
    let
        Source =
            #table(
                type table [Date Last Refreshed = datetime],
                {
                    {
                        DateTimeZone.UtcNow()
                    }
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                Source,
                {
                    {
                        "Date Last Refreshed",
                        type datetimezone
                    }
                }
            )
    in
        #"Changed Type";

shared base_build_scenario =
    let
        Source = base_build_index,
        #"Invoked Custom Function1" =
            Table.AddColumn(
                Source,
                "Scenario",
                each GetScenario([Full_Path])
            ),
        #"Removed Other Columns" =
            Table.SelectColumns(
                #"Invoked Custom Function1",
                {
                    "build_index",
                    "Scenario"
                }
            ),
        #"Expanded Scenario" =
            Table.ExpandTableColumn(
                #"Removed Other Columns",
                "Scenario",
                {
                    "description",
                    "scenario",
                    "start_time"
                },
                {
                    "description",
                    "scenario",
                    "start_time"
                }
            ),
        #"Duplicated Column" =
            Table.DuplicateColumn(
                #"Expanded Scenario",
                "scenario",
                "scenario - Copy"
            ),
        #"Merged Columns" =
            Table.CombineColumns(
                #"Duplicated Column",
                {
                    "scenario - Copy",
                    "description"
                },
                Combiner.CombineTextByDelimiter(
                    ":",
                    QuoteStyle.None
                ),
                "description"
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Merged Columns",
                {
                    {
                        "scenario",
                        Int64.Type
                    },
                    {
                        "start_time",
                        Int64.Type
                    },
                    {
                        "description",
                        type text
                    }
                }
            ),
        #"Inserted Merged Column" =
            Table.AddColumn(
                #"Changed Type",
                "bld_scn",
                each
                    Text.Combine(
                        {
                            Text.From([build_index], "en-AU"),
                            Text.From([scenario], "en-AU")
                        },
                        "_"
                    ),
                type text
            )
    //base_covid_region
    in
        #"Inserted Merged Column";

shared GetPowerbiOutputs =
    let
        Source2 =
            (pbidbpath as text) =>
                let
                    Source =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    powerbi_Table =
                        Source{[
                            Name = "powerbi_outputs",
                            Kind = "Table"
                        ]}
                            [Data],
                    FixBinary =
                        Table.TransformColumns(
                            powerbi_Table,
                            {
                                {
                                    "scenario",
                                    ConvertBinary,
                                    Int64.Type
                                }
                            }
                        ),
                    start_time = GetStartTime(pbidbpath),
                    #"Removed Columns" =
                        Table.RemoveColumns(
                            FixBinary,
                            {
                                "chain",
                                "run"
                            }
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            #"Removed Columns",
                            {
                                {
                                    "scenario",
                                    Int64.Type
                                }
                            }
                        ),
                    Replicate =
                        try
                            ReplicateBaseline(
                                #"Changed Type",
                                start_time
                            )
                        otherwise #"Changed Type",
                    #"Added Covid start date" =
                        Table.TransformColumns(
                            Replicate,
                            {
                                {
                                    "times",
                                    each _ + covid_date,
                                    Int64.Type
                                }
                            }
                        ),
                    #"Extracted Text After Delimiter" =
                        Table.TransformColumns(
                            #"Added Covid start date",
                            {
                                {
                                    "agegroup",
                                    each
                                        Text.AfterDelimiter(
                                            _,
                                            "_",
                                            {
                                                0,
                                                RelativePosition.FromEnd
                                            }
                                        ),
                                    type text
                                }
                            }
                        ),
                    #"Changed Type1" =
                        Table.TransformColumnTypes(
                            #"Extracted Text After Delimiter",
                            {
                                {
                                    "agegroup",
                                    Int64.Type
                                },
                                {
                                    "times",
                                    type date
                                }
                            }
                        ),
                    #"Renamed Columns" =
                        Table.RenameColumns(
                            #"Changed Type1",
                            {
                                {
                                    "times",
                                    "pbi_date"
                                },
                                {
                                    "scenario",
                                    "pbi_scenario"
                                },
                                {
                                    "value",
                                    "pbi_value"
                                },
                                {
                                    "agegroup",
                                    "dim_age"
                                },
                                {
                                    "clinical",
                                    "pbi_clinical"
                                },
                                {
                                    "compartment",
                                    "pbi_compartment"
                                }
                            }
                        ),
                    remove_clinical =
                        Table.TransformColumns(
                            #"Renamed Columns",
                            {
                                {
                                    "pbi_clinical",
                                    each Text.AfterDelimiter(_, "clinical_"),
                                    type text
                                }
                            }
                        ),
                    #"Replaced Value" =
                        Table.ReplaceValue(
                            remove_clinical,
                            null,
                            "-1",
                            Replacer.ReplaceValue,
                            {"pbi_clinical"}
                        )
                in
                    #"Replaced Value"
    in
        Source2;

shared base_derived_output =
    let
        Source = base_build_index,
        #"Invoked Custom Function" =
            Table.AddColumn(
                Source,
                "GetDerivedOutputs",
                each GetDerivedOutputs([Full_Path])
            ),
        #"Removed Other Columns1" =
            Table.SelectColumns(
                #"Invoked Custom Function",
                {
                    "build_index",
                    "GetDerivedOutputs"
                }
            ),
        #"Removed Errors" =
            Table.RemoveRowsWithErrors(
                #"Removed Other Columns1",
                {"GetDerivedOutputs"}
            ),
        #"Added Custom1" =
            List.Combine(
                Table.AddColumn(
                    #"Removed Errors",
                    "Custom",
                    (x) => Table.ColumnNames(x[GetDerivedOutputs])
                )
                    [Custom]
            ),
        columns_to_expand = List.Distinct(#"Added Custom1"),
        #"Expanded GetDerivedOutputs" =
            Table.ExpandTableColumn(
                #"Removed Errors",
                "GetDerivedOutputs",
                columns_to_expand
            ),
        #"Unpivoted Other Columns" =
            Table.UnpivotOtherColumns(
                #"Expanded GetDerivedOutputs",
                {
                    "build_index",
                    "scenario",
                    "times"
                },
                "Attribute",
                "Value"
            ),
        #"Renamed Columns" =
            Table.RenameColumns(
                #"Unpivoted Other Columns",
                {
                    {
                        "scenario",
                        "do_scenario"
                    },
                    {
                        "times",
                        "do_date"
                    },
                    {
                        "Attribute",
                        "stratification_name"
                    },
                    {
                        "Value",
                        "do_value"
                    }
                }
            ),
        //, "GetDerivedOutputs", {"do_scenario", "do_date", "stratification_name", "do_value"}, {"do_scenario", "do_date", "stratification_name", "do_value"}),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Renamed Columns",
                {
                    {
                        "do_scenario",
                        type text
                    },
                    {
                        "do_date",
                        type date
                    },
                    {
                        "stratification_name",
                        type text
                    },
                    {
                        "do_value",
                        type number
                    }
                }
            ),
        #"Inserted Merged Column" =
            Table.AddColumn(
                #"Changed Type",
                "bld_scn",
                each
                    Text.Combine(
                        {
                            Text.From([build_index], "en-AU"),
                            [do_scenario]
                        },
                        "_"
                    ),
                type text
            )
    in
        #"Inserted Merged Column";

shared GetDerivedOutputs =
    let
        Source =
            (pbidbpath as text) =>
                let
                    Source2 =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    derived_output_Table =
                        Source2{[
                            Name = "derived_outputs",
                            Kind = "Table"
                        ]}
                            [Data],
                    FixBinary =
                        Table.TransformColumns(
                            derived_output_Table,
                            {
                                {
                                    "scenario",
                                    ConvertBinary,
                                    Int64.Type
                                },
                                {
                                    "run",
                                    ConvertBinary,
                                    Int64.Type
                                }
                            }
                        ),
                    #"Changed Type1" =
                        Table.TransformColumnTypes(
                            FixBinary,
                            {
                                {
                                    "run",
                                    Int64.Type
                                },
                                {
                                    "scenario",
                                    Int64.Type
                                }
                            }
                        ),
                    start_time = GetStartTime(pbidbpath),
                    all_good =
                        Table.RemoveColumns(
                            #"Changed Type1",
                            {
                                "chain",
                                "run"
                            }
                        ),
                    Replicate = try ReplicateBaseline(all_good, start_time) otherwise all_good,
                    #"Added to Column" =
                        Table.TransformColumns(
                            Replicate,
                            {
                                {
                                    "times",
                                    each _ + covid_date,
                                    Int64.Type
                                }
                            }
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            #"Added to Column",
                            {
                                {
                                    "times",
                                    type date
                                }
                            }
                        )
                in
                    #"Changed Type"
    in
        Source;

shared base_uncertainty =
    let
        Source = base_build_index,
        #"Invoked Custom Function" =
            Table.AddColumn(
                Source,
                "GetUncertainty",
                each GetUncertainty([Full_Path])
            ),
        #"Removed Other Columns" =
            Table.SelectColumns(
                #"Invoked Custom Function",
                {
                    "build_index",
                    "GetUncertainty"
                }
            ),
        #"Removed Errors" =
            Table.RemoveRowsWithErrors(
                #"Removed Other Columns",
                {"GetUncertainty"}
            ),
        #"Expanded GetUncertainty" =
            Table.ExpandTableColumn(
                #"Removed Errors",
                "GetUncertainty",
                {
                    "un_scenario",
                    "un_date",
                    "un_quantile",
                    "stratification_id",
                    "un_value"
                },
                {
                    "un_scenario",
                    "un_date",
                    "un_quantile",
                    "stratification_id",
                    "un_value"
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Expanded GetUncertainty",
                {
                    {
                        "un_scenario",
                        type text
                    },
                    {
                        "un_quantile",
                        type number
                    },
                    {
                        "un_date",
                        type date
                    },
                    {
                        "un_value",
                        type number
                    },
                    {
                        "stratification_id",
                        Int64.Type
                    }
                }
            ),
        #"Inserted Merged Column" =
            Table.AddColumn(
                #"Changed Type",
                "bld_scn",
                each
                    Text.Combine(
                        {
                            Text.From([build_index], "en-AU"),
                            [un_scenario]
                        },
                        "_"
                    ),
                type text
            )
    in
        #"Inserted Merged Column";

shared GetUncertainty =
    let
        Source2 =
            (pbidbpath as text) =>
                let
                    Source =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    uncertainty_Table =
                        Source{[
                            Name = "uncertainty",
                            Kind = "Table"
                        ]}
                            [Data],
                    FixBinary =
                        Table.TransformColumns(
                            uncertainty_Table,
                            {
                                {
                                    "scenario",
                                    ConvertBinary,
                                    Int64.Type
                                }
                            }
                        ),
                    #"Renamed Columns1" =
                        Table.RenameColumns(
                            FixBinary,
                            {
                                {
                                    "time",
                                    "times"
                                }
                            }
                        ),
                    #"Changed Type1" =
                        Table.TransformColumnTypes(
                            #"Renamed Columns1",
                            {
                                {
                                    "scenario",
                                    Int64.Type
                                },
                                {
                                    "times",
                                    Int64.Type
                                }
                            }
                        ),
                    start_time = GetStartTime(pbidbpath),
                    Replicate =
                        try
                            ReplicateBaseline(
                                #"Changed Type1",
                                start_time
                            )
                        otherwise #"Changed Type1",
                    #"Added Covid start date" =
                        Table.TransformColumns(
                            Replicate,
                            {
                                {
                                    "times",
                                    each _ + covid_date,
                                    Int64.Type
                                }
                            }
                        ),
                    #"Rename Columns" =
                        Table.RenameColumns(
                            #"Added Covid start date",
                            {
                                {
                                    "scenario",
                                    "un_scenario"
                                },
                                {
                                    "quantile",
                                    "un_quantile"
                                },
                                {
                                    "times",
                                    "un_date"
                                },
                                {
                                    "type",
                                    "stratification_name"
                                },
                                {
                                    "value",
                                    "un_value"
                                }
                            }
                        ),
                    #"Merged Queries" =
                        Table.NestedJoin(
                            #"Rename Columns",
                            {"stratification_name"},
                            DimStratification,
                            {"stratification_name"},
                            "DimStratification",
                            JoinKind.LeftOuter
                        ),
                    #"Expanded DimStratification" =
                        Table.ExpandTableColumn(
                            #"Merged Queries",
                            "DimStratification",
                            {"stratification_id"},
                            {"stratification_id"}
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            #"Expanded DimStratification",
                            {
                                {
                                    "un_date",
                                    type date
                                }
                            }
                        ),
                    #"Removed Columns" =
                        Table.RemoveColumns(
                            #"Changed Type",
                            {"stratification_name"}
                        )
                in
                    #"Removed Columns"
    in
        Source2;

shared Calendar =
    let
        Source =
            List.Dates(
                StartDate,
                Length,
                #duration(1, 0, 0, 0)
            ),
        #"Converted to Table" =
            Table.FromList(
                Source,
                Splitter.SplitByNothing(),
                null,
                null,
                ExtraValues.Error
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Converted to Table",
                {
                    {
                        "Column1",
                        type date
                    }
                }
            ),
        #"Renamed Columns" =
            Table.RenameColumns(
                #"Changed Type",
                {
                    {
                        "Column1",
                        "cal_date"
                    }
                }
            ),
        Today = DateTime.Date(DateTime.LocalNow()),
        Length = Duration.Days(EndDate - StartDate),
        #"Inserted Year" =
            Table.AddColumn(
                #"Renamed Columns",
                "cal_year",
                each Date.Year([cal_date]),
                Int64.Type
            ),
        #"Inserted Month" =
            Table.AddColumn(
                #"Inserted Year",
                "cal_month",
                each Date.Month([cal_date]),
                Int64.Type
            ),
        #"Inserted Month Name" =
            Table.AddColumn(
                #"Inserted Month",
                "cal_month_name",
                each Date.MonthName([cal_date]),
                type text
            ),
        #"Added Custom Column" =
            Table.AddColumn(
                #"Inserted Month Name",
                "MonthYear",
                each
                    Text.Combine(
                        {
                            Text.Start([cal_month_name], 3),
                            "-",
                            Text.From([cal_year], "en-AU")
                        }
                    ),
                type text
            ),
        #"Inserted Addition" =
            Table.AddColumn(
                #"Added Custom Column",
                "cal_mon_yeat_int",
                each [cal_year] * 100 + [cal_month],
                Int64.Type
            )
    in
        #"Inserted Addition";

shared DimCompartment =
    let
        Source =
            #table(
                {"dim_compartment"},
                {
                    {
                        "susceptible"
                    },
                    {
                        "early_exposed"
                    },
                    {
                        "recovered"
                    },
                    {
                        "late_exposed"
                    },
                    {
                        "early_active"
                    },
                    {
                        "late_active"
                    },
                    {
                        "late"
                    },
                    {
                        "presympt"
                    }
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                Source,
                {
                    {
                        "dim_compartment",
                        type text
                    }
                }
            )
    in
        #"Changed Type";

shared DimAge =
    let
        age = {
            "00-04",
            "05-09",
            "10-14",
            "15-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75+"
        },
        #"Converted to Table1" =
            Table.FromList(
                age,
                Splitter.SplitByNothing(),
                null,
                null,
                ExtraValues.Error
            ),
        #"Duplicated Column" =
            Table.DuplicateColumn(
                #"Converted to Table1",
                "Column1",
                "Column1 - Copy"
            ),
        #"Extracted Text Before Delimiter" =
            Table.TransformColumns(
                #"Duplicated Column",
                {
                    {
                        "Column1",
                        each Text.BeforeDelimiter(_, "-"),
                        type text
                    }
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Extracted Text Before Delimiter",
                {
                    {
                        "Column1 - Copy",
                        type text
                    },
                    {
                        "Column1",
                        Int64.Type
                    }
                }
            ),
        #"Renamed Columns1" =
            Table.RenameColumns(
                #"Changed Type",
                {
                    {
                        "Column1",
                        "dim_age"
                    },
                    {
                        "Column1 - Copy",
                        "dim_age_group"
                    }
                }
            ),
        Custom3 =
            Table.InsertRows(
                #"Renamed Columns1",
                16,
                {
                    [
                        dim_age = -1,
                        dim_age_group = "unknown"
                    ]
                }
            ),
        Custom1 = Table.Buffer(Custom3)
    in
        Custom1;

shared DimScenario =
    let
        Source =
            #table(
                {"dim_scenario" as text},
                {
                    {
                        "S_0"
                    },
                    {
                        "S_1"
                    },
                    {
                        "S_2"
                    },
                    {
                        "S_3"
                    },
                    {
                        "S_4"
                    },
                    {
                        "S_5"
                    },
                    {
                        "S_6"
                    },
                    {
                        "S_7"
                    },
                    {
                        "S_8"
                    },
                    {
                        "S_9"
                    },
                    {
                        "S_10"
                    },
                    {
                        "S_11"
                    },
                    {
                        "S_12"
                    },
                    {
                        "S_13"
                    },
                    {
                        "S_14"
                    },
                    {
                        "S_15"
                    }
                }
            )
            as table,
        #"Added Index" =
            Table.AddIndexColumn(
                Source,
                "dim_scenario_id",
                0,
                1,
                Int64.Type
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Added Index",
                {
                    {
                        "dim_scenario",
                        type text
                    }
                }
            )
    in
        #"Changed Type";

shared DimClinical =
    let
        Source =
            #table(
                {"dim_clinical" as text},
                {
                    {
                        "non_sympt"
                    },
                    {
                        "sympt_non_hospital"
                    },
                    {
                        "sympt_isolate"
                    },
                    {
                        "hospital_non_icu"
                    },
                    {
                        "icu"
                    },
                    {
                        "-1"
                    }
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                Source,
                {
                    {
                        "dim_clinical",
                        type text
                    }
                }
            )
    in
        #"Changed Type";

shared DimRegion =
    let
        Source = base_build_index,
        #"Removed Other Columns" =
            Table.SelectColumns(
                Source,
                {"region_name"}
            ),
        #"Removed Duplicates" = Table.Distinct(#"Removed Other Columns"),
        #"Renamed Columns1" =
            Table.RenameColumns(
                #"Removed Duplicates",
                {
                    {
                        "region_name",
                        "dim_region"
                    }
                }
            ),
        #"Added Conditional Column" =
            Table.AddColumn(
                #"Renamed Columns1",
                "country",
                each
                    if [dim_region] = "national capital region" then
                        "philippines"
                    else if [dim_region] = "calabarzon" then
                        "philippines"
                    else if [dim_region] = "central visayas" then
                        "philippines"
                    else if [dim_region] = "davao city" then
                        "philippines"
                    else if [dim_region] = "davao region" then
                        "philippines"
                    else if [dim_region] = "victoria" then
                        "australia"
                    else if [dim_region] = "penang" then
                        "malaysia"
                    else if [dim_region] = "kuala lumpur" then
                        "malaysia"
                    else if [dim_region] = "johor" then
                        "malaysia"
                    else if [dim_region] = "selangor" then
                        "malaysia"
                    else
                        [dim_region],
                type text
            ),
        #"Renamed Columns" =
            Table.RenameColumns(
                #"Added Conditional Column",
                {
                    {
                        "country",
                        "dim_country"
                    }
                }
            ),
        #"Duplicated Column" =
            Table.DuplicateColumn(
                #"Renamed Columns",
                "dim_country",
                "dim_country_iso3"
            ),
        #"Replaced Value" =
            Table.ReplaceValue(
                #"Duplicated Column",
                "philippines",
                "PHL",
                Replacer.ReplaceText,
                {"dim_country_iso3"}
            ),
        #"Replaced Value1" =
            Table.ReplaceValue(
                #"Replaced Value",
                "malaysia",
                "MYS",
                Replacer.ReplaceText,
                {"dim_country_iso3"}
            ),
        #"Replaced Value2" =
            Table.ReplaceValue(
                #"Replaced Value1",
                "nepal",
                "NPL",
                Replacer.ReplaceText,
                {"dim_country_iso3"}
            ),
        #"Replaced Value3" =
            Table.ReplaceValue(
                #"Replaced Value2",
                "sri_lanka",
                "LKA",
                Replacer.ReplaceText,
                {"dim_country_iso3"}
            )
    in
        #"Replaced Value3";

shared #"Key Measures" =
    let
        Source =
            #table(
                {"Measures"},
                {
                    {
                        0
                    }
                }
            ),
        #"Removed Columns" =
            Table.RemoveColumns(
                Source,
                {"Measures"}
            )
    in
        #"Removed Columns";

shared social_mixing =
    let
        Source = PowerBI.Dataflows(null),
        #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e" = Source{[workspaceId = "d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"]}[Data],
        #"54e82e7b-6c12-4f7d-a631-69cd553c9d59" = #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"{[dataflowId = "54e82e7b-6c12-4f7d-a631-69cd553c9d59"]}[Data],
        social_mixing1 = #"54e82e7b-6c12-4f7d-a631-69cd553c9d59"{[entity = "social_mixing"]}[Data],
        #"Replaced Value" =
            Table.ReplaceValue(
                social_mixing1,
                "Sri Lanka",
                "sri_lanka",
                Replacer.ReplaceText,
                {"country"}
            )
    in
        #"Replaced Value";

shared google_mobility =
    let
        Source = PowerBI.Dataflows(null),
        #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e" = Source{[workspaceId = "d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"]}[Data],
        #"54e82e7b-6c12-4f7d-a631-69cd553c9d59" = #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"{[dataflowId = "54e82e7b-6c12-4f7d-a631-69cd553c9d59"]}[Data],
        google_mobility1 = #"54e82e7b-6c12-4f7d-a631-69cd553c9d59"{[entity = "google_mobility"]}[Data],
        #"Replaced Value" =
            Table.ReplaceValue(
                google_mobility1,
                "National Capital Region",
                "national capital region",
                Replacer.ReplaceText,
                {"sub_region_1"}
            ),
        #"Replaced Value1" =
            Table.ReplaceValue(
                #"Replaced Value",
                "Central Visayas",
                "central visayas",
                Replacer.ReplaceText,
                {"sub_region_1"}
            ),
        #"Replaced Value2" =
            Table.ReplaceValue(
                #"Replaced Value1",
                "Sri Lanka",
                "sri_lanka",
                Replacer.ReplaceText,
                {"sub_region_1"}
            )
    in
        #"Replaced Value2";

shared base_calibration =
    let
        Source = base_build_index,
        #"Invoked Custom Function" =
            Table.AddColumn(
                Source,
                "GetCalibrationValues",
                each
                    GetCalibrationValues(
                        [Full_Path],
                        "target"
                    )
            ),
        #"Removed Other Columns" =
            Table.SelectColumns(
                #"Invoked Custom Function",
                {
                    "build_index",
                    "GetCalibrationValues"
                }
            ),
        #"Removed Errors" =
            Table.RemoveRowsWithErrors(
                #"Removed Other Columns",
                {"GetCalibrationValues"}
            ),
        #"Expanded GetCalibrationValues" =
            Table.ExpandTableColumn(
                #"Removed Errors",
                "GetCalibrationValues",
                {
                    "stratification_id",
                    "cal_date",
                    "cal_value"
                },
                {
                    "stratification_id",
                    "cal_date",
                    "cal_value"
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Expanded GetCalibrationValues",
                {
                    {
                        "cal_date",
                        type date
                    },
                    {
                        "cal_value",
                        Int64.Type
                    },
                    {
                        "stratification_id",
                        Int64.Type
                    }
                }
            )
    in
        #"Changed Type";

shared base_stratification =
    let
        Source = base_build_index,
        #"Invoked Custom Function" =
            Table.AddColumn(
                Source,
                "StratificationNames",
                each GetColumnNames([Full_Path])
            )
                [StratificationNames],
        Custom1 = List.Distinct(List.Combine(#"Invoked Custom Function")),
        #"Converted to Table" =
            Table.FromList(
                Custom1,
                Splitter.SplitByNothing(),
                null,
                null,
                ExtraValues.Error
            ),
        #"Renamed Columns" =
            Table.RenameColumns(
                #"Converted to Table",
                {
                    {
                        "Column1",
                        "stratification_name"
                    }
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Renamed Columns",
                {
                    {
                        "stratification_name",
                        type text
                    }
                }
            ),
        #"Added Index" =
            Table.AddIndexColumn(
                #"Changed Type",
                "stratification_id",
                0,
                1,
                Int64.Type
            ),
        #"Inserted Text Between Delimiters" =
            Table.AddColumn(
                #"Added Index",
                "stratification_age",
                each
                    Number.FromText(
                        Text.BetweenDelimiters(
                            [stratification_name],
                            "agegroup_",
                            "X"
                        )
                    ),
                Int64.Type
            ),
        #"Inserted Text After Delimiter" =
            Table.AddColumn(
                #"Inserted Text Between Delimiters",
                "stratification_clinical",
                each
                    Text.AfterDelimiter(
                        [stratification_name],
                        "clinical_"
                    ),
                type text
            ),
        #"Replaced Value" =
            Table.ReplaceValue(
                #"Inserted Text After Delimiter",
                "",
                "-1",
                Replacer.ReplaceValue,
                {"stratification_clinical"}
            ),
        #"Inserted Text Before Delimiter" =
            Table.AddColumn(
                #"Replaced Value",
                "stratification_strata1",
                each
                    if [stratification_age] <> null then
                        Text.BeforeDelimiter(
                            [stratification_name],
                            "X"
                        )
                    else
                        null,
                type text
            ),
        #"Replaced Value1" =
            Table.ReplaceValue(
                #"Inserted Text Before Delimiter",
                null,
                -1,
                Replacer.ReplaceValue,
                {"stratification_age"}
            ),
        #"Replaced Value2" =
            Table.ReplaceValue(
                #"Replaced Value1",
                null,
                each [stratification_name],
                Replacer.ReplaceValue,
                {"stratification_strata1"}
            ),
        #"Changed Type2" =
            Table.TransformColumnTypes(
                #"Replaced Value2",
                {
                    {
                        "stratification_strata1",
                        type text
                    }
                }
            )
    in
        #"Changed Type2";

shared derived_output =
    let
        Source = base_derived_output,
        Custom1 =
            Table.TransformColumns(
                Source,
                {
                    {
                        "stratification_name",
                        each DimStratification{[stratification_name = _]}[stratification_id],
                        Int64.Type
                    }
                }
            ),
        #"Renamed Columns" =
            Table.RenameColumns(
                Custom1,
                {
                    {
                        "stratification_name",
                        "stratification_id"
                    }
                }
            )
    in
        #"Renamed Columns";

shared GetCalibrationValues =
    let
        Func =
            (pbidbpath as text, table_name as text) =>
                let
                    Source =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    targets_Table =
                        Source{[
                            Name = "targets",
                            Kind = "Table"
                        ]}
                            [Data],
                    FixBinary =
                        Table.TransformColumns(
                            targets_Table,
                            {
                                {
                                    "times",
                                    ConvertBinary,
                                    Int64.Type
                                }
                            }
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            FixBinary,
                            {
                                {
                                    "times",
                                    Int64.Type
                                }
                            }
                        ),
                    Transform_columns =
                        Table.TransformColumns(
                            #"Changed Type",
                            {
                                {
                                    "times",
                                    each _ + covid_date,
                                    Int64.Type
                                },
                                {
                                    "key",
                                    each DimStratification{[stratification_name = _]}[stratification_id],
                                    Int64.Type
                                }
                            }
                        ),
                    #"Changed Type2" =
                        Table.TransformColumnTypes(
                            Transform_columns,
                            {
                                {
                                    "times",
                                    type date
                                }
                            }
                        ),
                    #"Renamed Columns" =
                        Table.RenameColumns(
                            #"Changed Type2",
                            {
                                {
                                    "key",
                                    "stratification_id"
                                },
                                {
                                    "times",
                                    "cal_date"
                                },
                                {
                                    "value",
                                    "cal_value"
                                }
                            }
                        )
                in
                    #"Renamed Columns"
    in
        Func;

shared build_index =
    let
        Source = base_build_index,
        #"Inserted Merged Column" =
            Table.AddColumn(
                Source,
                "Model",
                each
                    Text.Combine(
                        {
                            [region_name],
                            Text.From([build_date], "en-AU")
                        },
                        "-"
                    ),
                type text
            )
    in
        #"Inserted Merged Column";

shared powerbi_output =
    let
        Source = base_powerbi_output
    in
        Source;

shared uncertainty =
    let
        Source = base_uncertainty
    in
        Source;

shared calibration_values =
    let
        Source = base_calibration
    in
        Source;

shared DimStratification =
    let
        Source = Table.Buffer(base_stratification)
    in
        Source;

shared GetMCMC_param =
    let
        Source =
            (pbidbpath as text) =>
                let
                    Source =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    powerbi_outputs_Table =
                        Source{[
                            Name = "mcmc_params",
                            Kind = "Table"
                        ]}
                            [Data],
                    fix_chain =
                        Table.TransformColumns(
                            powerbi_outputs_Table,
                            {
                                "chain",
                                each Lines.FromBinary(_, null, null, 1252){0},
                                type number
                            }
                        ),
                    fix_run =
                        Table.TransformColumns(
                            fix_chain,
                            {
                                "run",
                                each Lines.FromBinary(_, null, null, 1252){0},
                                type number
                            }
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            fix_run,
                            {
                                {
                                    "chain",
                                    Int64.Type
                                },
                                {
                                    "run",
                                    Int64.Type
                                }
                            }
                        ),
                    #"Renamed Columns1" =
                        Table.RenameColumns(
                            #"Changed Type",
                            {
                                {
                                    "chain",
                                    "mcmc_chain"
                                },
                                {
                                    "run",
                                    "mcmc_run"
                                },
                                {
                                    "name",
                                    "mcmc_parameter_name"
                                },
                                {
                                    "value",
                                    "mcmc_parameter_value"
                                }
                            }
                        )
                in
                    #"Renamed Columns1"
    in
        Source;

shared base_mcmc_param =
    let
        Source = base_build_index,
        #"Invoked Custom Function" =
            Table.AddColumn(
                Source,
                "GetMCMC",
                each GetMCMC_param([Full_Path])
            ),
        #"Removed Other Columns" =
            Table.SelectColumns(
                #"Invoked Custom Function",
                {
                    "build_index",
                    "GetMCMC"
                }
            ),
        #"Removed Errors" =
            Table.RemoveRowsWithErrors(
                #"Removed Other Columns",
                {"GetMCMC"}
            ),
        #"Expanded GetMCMC" =
            Table.ExpandTableColumn(
                #"Removed Errors",
                "GetMCMC",
                {
                    "mcmc_chain",
                    "mcmc_run",
                    "mcmc_parameter_name",
                    "mcmc_parameter_value"
                },
                {
                    "mcmc_chain",
                    "mcmc_run",
                    "mcmc_parameter_name",
                    "mcmc_parameter_value"
                }
            ),
        #"Changed Type1" =
            Table.TransformColumnTypes(
                #"Expanded GetMCMC",
                {
                    {
                        "mcmc_chain",
                        Int64.Type
                    },
                    {
                        "mcmc_run",
                        Int64.Type
                    },
                    {
                        "mcmc_parameter_name",
                        type text
                    },
                    {
                        "mcmc_parameter_value",
                        type number
                    }
                }
            )
    in
        #"Changed Type1";

shared base_mcmc_run =
    let
        Source = base_build_index,
        #"Invoked Custom Function" =
            Table.AddColumn(
                Source,
                "GetMCMC_run",
                each GetMCMC_run([Full_Path])
            ),
        #"Removed Other Columns" =
            Table.SelectColumns(
                #"Invoked Custom Function",
                {
                    "build_index",
                    "GetMCMC_run"
                }
            ),
        #"Removed Errors" =
            Table.RemoveRowsWithErrors(
                #"Removed Other Columns",
                {"GetMCMC_run"}
            ),
        #"Expanded GetMCMC_run" =
            Table.ExpandTableColumn(
                #"Removed Errors",
                "GetMCMC_run",
                {
                    "mcmc_chain",
                    "mcmc_run",
                    "mcmc_loglikelihood",
                    "mcmc_ap_loglikelihood",
                    "mcmc_accept",
                    "mcmc_weight"
                },
                {
                    "mcmc_chain",
                    "mcmc_run",
                    "mcmc_loglikelihood",
                    "mcmc_ap_loglikelihood",
                    "mcmc_accept",
                    "mcmc_weight"
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Expanded GetMCMC_run",
                {
                    {
                        "mcmc_chain",
                        Int64.Type
                    },
                    {
                        "mcmc_run",
                        Int64.Type
                    },
                    {
                        "mcmc_accept",
                        Int64.Type
                    },
                    {
                        "mcmc_weight",
                        Int64.Type
                    },
                    {
                        "mcmc_loglikelihood",
                        type number
                    },
                    {
                        "mcmc_ap_loglikelihood",
                        type number
                    }
                }
            )
    in
        #"Changed Type";

shared base_parameter =
    let
        Source = base_mcmc_param,
        #"Removed Other Columns" =
            Table.SelectColumns(
                Source,
                {"mcmc_parameter_name"}
            ),
        #"Removed Duplicates" = Table.Distinct(#"Removed Other Columns"),
        #"Added Index" =
            Table.AddIndexColumn(
                #"Removed Duplicates",
                "parameter_id",
                0,
                1,
                Int64.Type
            )
    in
        #"Added Index";

shared parameter =
    let
        Source = base_parameter
    in
        Source;

shared covid_date =
    let
        Source = Int64.From(43830)
    in
        Source;

shared GetMCMC_run =
    let
        Source =
            (pbidbpath as text) =>
                let
                    Source =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    powerbi_outputs_Table =
                        Source{[
                            Name = "mcmc_run",
                            Kind = "Table"
                        ]}
                            [Data],
                    fix_chain =
                        Table.TransformColumns(
                            powerbi_outputs_Table,
                            {
                                "chain",
                                each Lines.FromBinary(_, null, null, 1252){0},
                                type number
                            }
                        ),
                    fix_run =
                        Table.TransformColumns(
                            fix_chain,
                            {
                                "run",
                                each Lines.FromBinary(_, null, null, 1252){0},
                                type number
                            }
                        ),
                    fix_accept =
                        Table.TransformColumns(
                            fix_run,
                            {
                                "accept",
                                each Lines.FromBinary(_, null, null, 1252){0},
                                type number
                            }
                        ),
                    fix_weight =
                        Table.TransformColumns(
                            fix_accept,
                            {
                                "weight",
                                each Lines.FromBinary(_, null, null, 1252){0},
                                type number
                            }
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            fix_weight,
                            {
                                {
                                    "chain",
                                    Int64.Type
                                },
                                {
                                    "run",
                                    Int64.Type
                                },
                                {
                                    "accept",
                                    Int64.Type
                                },
                                {
                                    "weight",
                                    Int64.Type
                                }
                            }
                        ),
                    #"Renamed Columns" =
                        Table.RenameColumns(
                            #"Changed Type",
                            {
                                {
                                    "chain",
                                    "mcmc_chain"
                                },
                                {
                                    "run",
                                    "mcmc_run"
                                },
                                {
                                    "loglikelihood",
                                    "mcmc_loglikelihood"
                                },
                                {
                                    "ap_loglikelihood",
                                    "mcmc_ap_loglikelihood"
                                },
                                {
                                    "accept",
                                    "mcmc_accept"
                                },
                                {
                                    "weight",
                                    "mcmc_weight"
                                }
                            }
                        )
                in
                    #"Renamed Columns"
    in
        Source;

shared Scale =
    let
        Source =
            #table(
                {"Scale type"},
                {
                    {
                        "Linear"
                    },
                    {
                        "Log"
                    }
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                Source,
                {
                    {
                        "Scale type",
                        type text
                    }
                }
            )
    in
        #"Changed Type";

shared StartDate =
    #date(2020, 1, 1)
    meta
    [
        IsParameterQuery = true,
        Type = "Date",
        IsParameterQueryRequired = true
    ];

shared EndDate =
    #date(2022, 1, 1)
    meta
    [
        IsParameterQuery = true,
        Type = "Date",
        IsParameterQueryRequired = true
    ];

shared mcmc =
    let
        Source =
            Table.NestedJoin(
                base_mcmc_run,
                {
                    "build_index",
                    "mcmc_chain",
                    "mcmc_run"
                },
                base_mcmc_param,
                {
                    "build_index",
                    "mcmc_chain",
                    "mcmc_run"
                },
                "base_mcmc_param",
                JoinKind.RightOuter
            ),
        #"Expanded base_mcmc_param" =
            Table.ExpandTableColumn(
                Source,
                "base_mcmc_param",
                {
                    "mcmc_parameter_name",
                    "mcmc_parameter_value"
                },
                {
                    "mcmc_parameter_name",
                    "mcmc_parameter_value"
                }
            )
    in
        #"Expanded base_mcmc_param";

shared build_scenario =
    let
        Source = base_build_scenario,
        #"Removed Errors" =
            Table.RemoveRowsWithErrors(
                Source,
                {"scenario"}
            )
    in
        #"Removed Errors";

shared sanddance =
    let
        Source = mcmc,
        #"Merged Queries" =
            Table.NestedJoin(
                Source,
                {
                    "build_index",
                    "mcmc_chain",
                    "mcmc_run"
                },
                base_mcmc_param,
                {
                    "build_index",
                    "mcmc_chain",
                    "mcmc_run"
                },
                "base_mcmc_param",
                JoinKind.LeftOuter
            ),
        #"Expanded base_mcmc_param" =
            Table.ExpandTableColumn(
                #"Merged Queries",
                "base_mcmc_param",
                {
                    "mcmc_parameter_name",
                    "mcmc_parameter_value"
                },
                {
                    "mcmc_parameter_name.1",
                    "mcmc_parameter_value.1"
                }
            )
    in
        #"Expanded base_mcmc_param";

shared data_location =
    "M:\Documents\@Projects\Covid_consolidate\output"
    meta
    [
        IsParameterQuery = true,
        List = {
            "M:\Documents\@Projects\Covid_consolidate\output",
            "M:\Documents\@Projects\Covid_consolidate\output_fixed",
            "M:\Documents\@Projects\Covid_consolidate\output_test",
            "M:\Documents\@Projects\Covid_consolidate\output_pivot",
            "C:\Users\kogmaw\Desktop\output"
        },
        DefaultValue = ...,
        Type = "Text",
        IsParameterQueryRequired = false
    ];

shared GetBuild =
    let
        Source =
            (pbidbpath as text) =>
                let
                    //Source = Odbc.Query("database="&pbidbpath2&";dsn=SQLite3 Datasource", "select * #(lf)from build"),
                    Source =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    build_Table =
                        Source{[
                            Name = "build",
                            Kind = "Table"
                        ]}
                            [Data],
                    #"Split Column by Delimiter" =
                        Table.SplitColumn(
                            build_Table,
                            "build_key",
                            Splitter.SplitTextByDelimiter(
                                "-",
                                QuoteStyle.Csv
                            ),
                            {
                                "unix_epoch_time",
                                "git_commit"
                            }
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            #"Split Column by Delimiter",
                            {
                                {
                                    "unix_epoch_time",
                                    Int64.Type
                                }
                            }
                        ),
                    #"Inserted Division" =
                        Table.AddColumn(
                            #"Changed Type",
                            "build_time",
                            each 25569 + ([unix_epoch_time] / (60 * 60 * 24)),
                            type number
                        ),
                    #"Changed Type1" =
                        Table.TransformColumnTypes(
                            #"Inserted Division",
                            {
                                {
                                    "build_time",
                                    type datetime
                                }
                            }
                        ),
                    #"Inserted Date" =
                        Table.AddColumn(
                            #"Changed Type1",
                            "build_date",
                            each DateTime.Date([build_time]),
                            type date
                        )
                in
                    #"Inserted Date"
    in
        Source;

shared GetScenario =
    let
        Source2 =
            (pbidbpath as text) =>
                let
                    Source =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    scenario_Table =
                        Source{[
                            Name = "scenario",
                            Kind = "Table"
                        ]}
                            [Data],
                    FixBinary =
                        Table.TransformColumns(
                            scenario_Table,
                            {
                                {
                                    "scenario",
                                    ConvertBinary,
                                    Int64.Type
                                },
                                {
                                    "start_time",
                                    ConvertBinary,
                                    Int64.Type
                                }
                            }
                        )
                in
                    FixBinary
    in
        Source2;

shared pbidbpath2 =
    null
    meta
    [
        IsParameterQuery = true,
        Type = "Text",
        IsParameterQueryRequired = false
    ];

shared GetStartTime =
    let
        Source =
            (pbidbpath as text) =>
                let
                    Source2 =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    scenario_Table =
                        Source2{[
                            Name = "scenario",
                            Kind = "Table"
                        ]}
                            [Data],
                    FixBinary =
                        Table.TransformColumns(
                            scenario_Table,
                            {
                                {
                                    "scenario",
                                    ConvertBinary,
                                    Int64.Type
                                },
                                {
                                    "start_time",
                                    ConvertBinary,
                                    Int64.Type
                                }
                            }
                        ),
                    #"Changed Type" =
                        Table.TransformColumnTypes(
                            FixBinary,
                            {
                                {
                                    "scenario",
                                    Int64.Type
                                },
                                {
                                    "start_time",
                                    Int64.Type
                                }
                            }
                        ),
                    #"Filtered Rows" =
                        Table.SelectRows(
                            #"Changed Type",
                            each ([scenario] <> 0)
                        )
                            [[scenario], [start_time]]
                in
                    #"Filtered Rows"
    in
        Source;

shared ReplicateBaseline =
    (a_table as table, start_table as table) as table =>
        let
            #"Changed Type" =
                Table.TransformColumnTypes(
                    a_table,
                    {
                        {
                            "scenario",
                            Int64.Type
                        }
                    }
                ),
            buffer_table = Table.Buffer(#"Changed Type"),
            #"Added Custom" =
                Table.AddColumn(
                    start_table,
                    "Custom",
                    (x) =>
                        Table.SelectRows(
                            buffer_table,
                            each
                                [times]
                                < x[start_time]
                                and [scenario]
                                = 0
                        )
                )
                    [[scenario], [Custom]],
            all_columns = Table.ColumnNames(buffer_table),
            col_to_expand =
                List.RemoveMatchingItems(
                    all_columns,
                    {"scenario"}
                ),
            exapand_table =
                Table.ExpandTableColumn(
                    #"Added Custom",
                    "Custom",
                    col_to_expand
                ),
            all_done =
                try
                    Table.Combine(
                        {
                            a_table,
                            exapand_table
                        }
                    )
                otherwise a_table
        in
            all_done;

shared FixInterventionStartValues =
    let
        Source =
            (all_good as table, start_time as number) as table =>
                let
                    scenario_0_values =
                        Table.SelectRows(
                            all_good,
                            each [times] = start_time and [scenario] = 0
                        ),
                    distinct_scenarios = Table.Distinct(all_good[[scenario]]),
                    #"Added Custom" =
                        Table.AddColumn(
                            distinct_scenarios,
                            "scenario.1",
                            each 0,
                            Int64.Type
                        ),
                    #"Merged Queries" =
                        Table.NestedJoin(
                            #"Added Custom",
                            {"scenario.1"},
                            scenario_0_values,
                            {"scenario"},
                            "start_values",
                            JoinKind.LeftOuter
                        )
                            [[scenario], [start_values]],
                    columns_to_expand =
                        List.RemoveMatchingItems(
                            Table.ColumnNames(scenario_0_values),
                            {"scenario"}
                        ),
                    exapand_table =
                        Table.ExpandTableColumn(
                            #"Merged Queries",
                            "start_values",
                            columns_to_expand
                        ),
                    table_without_start_time =
                        Table.SelectRows(
                            all_good,
                            each [times] <> start_time
                        ),
                    Custom1 =
                        Table.Combine(
                            {
                                exapand_table,
                                table_without_start_time
                            }
                        )
                in
                    Custom1
    in
        Source;

shared ConvertBinary =
    let
        Source = (_) => try Lines.FromBinary(_, null, null, 1252){0} otherwise _
    in
        Source;

shared DimWith =
    let
        Source =
            List.Generate(
                () => 0,
                each _ <= 75,
                each _ + 5
            ),
        #"Converted to Table" =
            Table.FromList(
                Source,
                Splitter.SplitByNothing(),
                null,
                null,
                ExtraValues.Error
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Converted to Table",
                {
                    {
                        "Column1",
                        Int64.Type
                    }
                }
            ),
        #"Renamed Columns" =
            Table.RenameColumns(
                #"Changed Type",
                {
                    {
                        "Column1",
                        "With"
                    }
                }
            ),
        #"Added Conditional Column" =
            Table.AddColumn(
                #"Renamed Columns",
                "dim_age_group",
                each
                    if [With] = 0 then
                        "00-04"
                    else if [With] = 5 then
                        "05-09"
                    else if [With] = 10 then
                        "10-14"
                    else if [With] = 15 then
                        "15-19"
                    else if [With] = 20 then
                        "20-24"
                    else if [With] = 25 then
                        "25-29"
                    else if [With] = 30 then
                        "30-34"
                    else if [With] = 35 then
                        "35-39"
                    else if [With] = 40 then
                        "40-44"
                    else if [With] = 45 then
                        "45-49"
                    else if [With] = 50 then
                        "50-54"
                    else if [With] = 55 then
                        "55-59"
                    else if [With] = 60 then
                        "60-64"
                    else if [With] = 65 then
                        "65-69"
                    else if [With] = 70 then
                        "70-74"
                    else if [With] = 75 then
                        "75+"
                    else
                        null,
                type text
            )
    in
        #"Added Conditional Column";

shared GetColumnNames =
    let
        Source =
            (pbidbpath as text) =>
                let
                    Source2 =
                        Odbc.DataSource(
                            "database="
                            & pbidbpath
                            & ";dsn=SQLite3 Datasource",
                            [HierarchicalNavigation = true]
                        ),
                    derived_output_Table =
                        Source2{[
                            Name = "derived_outputs",
                            Kind = "Table"
                        ]}
                            [Data],
                    Custom1 =
                        List.RemoveMatchingItems(
                            Table.ColumnNames(derived_output_Table),
                            {
                                "chain",
                                "run",
                                "scenario",
                                "times"
                            }
                        )
                in
                    Custom1
    in
        Source;

shared #"owid-covid-latest" =
    let
        Source =
            Csv.Document(
                Web.Contents("https://covid.ourworldindata.org/data/owid-covid-data.csv"),
                [
                    Delimiter = ",",
                    Columns = 60,
                    Encoding = 65001,
                    QuoteStyle = QuoteStyle.None
                ]
            ),
        #"Promoted Headers" =
            Table.PromoteHeaders(
                Source,
                [PromoteAllScalars = true]
            ),
        #"Filtered Rows" =
            Table.SelectRows(
                #"Promoted Headers",
                each
                    (
                        [iso_code]
                        = "AUS"
                        or [iso_code]
                        = "IDN"
                        or [iso_code]
                        = "LKA"
                        or [iso_code]
                        = "MYS"
                        or [iso_code]
                        = "NPL"
                        or [iso_code]
                        = "PHL"
                    )
            )
    in
        #"Filtered Rows";

shared movement_range_lka =
    let
        Source =
            Excel.Workbook(
                File.Contents("C:\Users\kogmaw\Desktop\mobility\LKA.xlsx"),
                null,
                true
            ),
        test_Sheet =
            Source{[
                Item = "test",
                Kind = "Sheet"
            ]}
                [Data],
        #"Promoted Headers" =
            Table.PromoteHeaders(
                test_Sheet,
                [PromoteAllScalars = true]
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Promoted Headers",
                {
                    {
                        "GID_0",
                        type text
                    },
                    {
                        "NAME_0",
                        type text
                    },
                    {
                        "GID_1",
                        type text
                    },
                    {
                        "NAME_1",
                        type text
                    },
                    {
                        "NL_NAME_1",
                        type text
                    },
                    {
                        "GID_2",
                        type text
                    },
                    {
                        "NAME_2",
                        type text
                    },
                    {
                        "VARNAME_2",
                        type text
                    },
                    {
                        "NL_NAME_2",
                        type text
                    },
                    {
                        "TYPE_2",
                        type text
                    },
                    {
                        "ENGTYPE_2",
                        type text
                    },
                    {
                        "CC_2",
                        type text
                    },
                    {
                        "HASC_2",
                        type text
                    },
                    {
                        "ds",
                        type date
                    },
                    {
                        "country",
                        type text
                    },
                    {
                        "polygon_source",
                        type text
                    },
                    {
                        "polygon_id",
                        type text
                    },
                    {
                        "polygon_name",
                        type text
                    },
                    {
                        "baseline_name",
                        type text
                    },
                    {
                        "baseline_type",
                        type text
                    },
                    {
                        "all_day_bing_tiles_visited_relative_change",
                        type number
                    },
                    {
                        "all_day_ratio_single_tile_users",
                        type number
                    },
                    {
                        "total_pop_sum",
                        type number
                    },
                    {
                        "elderly_60_plus_sum",
                        type number
                    },
                    {
                        "Percent_Elderly",
                        type number
                    },
                    {
                        "risk_score",
                        Int64.Type
                    },
                    {
                        "scaled_score",
                        Int64.Type
                    }
                }
            )
    in
        #"Changed Type";

shared gadm36_LKA =
    let
        Source = Json.Document(File.Contents("C:\Users\kogmaw\Desktop\mobility\gadm36_LKA.json")),
        #"Converted to Table" = Table.FromRecords({Source}),
        #"Expanded objects" =
            Table.ExpandRecordColumn(
                #"Converted to Table",
                "objects",
                {
                    "gadm36_LKA_2",
                    "gadm36_LKA_1",
                    "gadm36_LKA_0"
                },
                {
                    "gadm36_LKA_2",
                    "gadm36_LKA_1",
                    "gadm36_LKA_0"
                }
            ),
        #"Expanded gadm36_LKA_2" =
            Table.ExpandRecordColumn(
                #"Expanded objects",
                "gadm36_LKA_2",
                {
                    "type",
                    "geometries"
                },
                {
                    "type.1",
                    "geometries"
                }
            ),
        #"Removed Other Columns" =
            Table.SelectColumns(
                #"Expanded gadm36_LKA_2",
                {"geometries"}
            ),
        #"Expanded geometries" =
            Table.ExpandListColumn(
                #"Removed Other Columns",
                "geometries"
            ),
        #"Expanded geometries1" =
            Table.ExpandRecordColumn(
                #"Expanded geometries",
                "geometries",
                {
                    "arcs",
                    "type",
                    "properties"
                },
                {
                    "arcs",
                    "type",
                    "properties"
                }
            ),
        #"Removed Other Columns1" =
            Table.SelectColumns(
                #"Expanded geometries1",
                {"properties"}
            ),
        #"Expanded properties" =
            Table.ExpandRecordColumn(
                #"Removed Other Columns1",
                "properties",
                {
                    "GID_0",
                    "NAME_0",
                    "GID_1",
                    "NAME_1",
                    "NL_NAME_1",
                    "GID_2",
                    "NAME_2",
                    "VARNAME_2",
                    "NL_NAME_2",
                    "TYPE_2",
                    "ENGTYPE_2",
                    "CC_2",
                    "HASC_2"
                },
                {
                    "GID_0",
                    "NAME_0",
                    "GID_1",
                    "NAME_1",
                    "NL_NAME_1",
                    "GID_2",
                    "NAME_2",
                    "VARNAME_2",
                    "NL_NAME_2",
                    "TYPE_2",
                    "ENGTYPE_2",
                    "CC_2",
                    "HASC_2"
                }
            ),
        #"Changed Type" =
            Table.TransformColumnTypes(
                #"Expanded properties",
                {
                    {
                        "GID_0",
                        type text
                    },
                    {
                        "NAME_0",
                        type text
                    },
                    {
                        "GID_1",
                        type text
                    },
                    {
                        "NAME_1",
                        type text
                    },
                    {
                        "NL_NAME_1",
                        type text
                    },
                    {
                        "GID_2",
                        type text
                    },
                    {
                        "NAME_2",
                        type text
                    },
                    {
                        "VARNAME_2",
                        type text
                    },
                    {
                        "NL_NAME_2",
                        type text
                    },
                    {
                        "TYPE_2",
                        type text
                    },
                    {
                        "ENGTYPE_2",
                        type text
                    },
                    {
                        "CC_2",
                        type text
                    },
                    {
                        "HASC_2",
                        type text
                    }
                }
            )
    in
        #"Changed Type";

shared movement_range =
    let
        Source = PowerBI.Dataflows(null),
        #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e" = Source{[workspaceId = "d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"]}[Data],
        #"54e82e7b-6c12-4f7d-a631-69cd553c9d59" = #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"{[dataflowId = "54e82e7b-6c12-4f7d-a631-69cd553c9d59"]}[Data],
        facebook_movement1 = #"54e82e7b-6c12-4f7d-a631-69cd553c9d59"{[entity = "facebook_movement"]}[Data],
        #"Filtered Rows" =
            Table.SelectRows(
                facebook_movement1,
                each
                    ([country] = "LKA" or [country] = "PHL")
                    or [country]
                    = "MYS"
            ),
        #"Merged Queries" =
            Table.NestedJoin(
                #"Filtered Rows",
                {
                    "country",
                    "polygon_id"
                },
                gadm36_LKA,
                {
                    "GID_0",
                    "GID_2"
                },
                "gadm36_LKA",
                JoinKind.LeftOuter
            ),
        #"Expanded gadm36_LKA" =
            Table.ExpandTableColumn(
                #"Merged Queries",
                "gadm36_LKA",
                {
                    "NAME_1",
                    "NAME_2"
                },
                {
                    "NAME_1",
                    "NAME_2"
                }
            ),
        #"Filtered Rows1" =
            Table.SelectRows(
                #"Expanded gadm36_LKA",
                each true
            )
    in
        #"Filtered Rows1";

shared facebook_movement =
    let
        Source = PowerBI.Dataflows(null),
        #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e" = Source{[workspaceId = "d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"]}[Data],
        #"54e82e7b-6c12-4f7d-a631-69cd553c9d59" = #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"{[dataflowId = "54e82e7b-6c12-4f7d-a631-69cd553c9d59"]}[Data],
        facebook_movement1 = #"54e82e7b-6c12-4f7d-a631-69cd553c9d59"{[entity = "facebook_movement"]}[Data]
    in
        facebook_movement1;

shared PHSM =
    let
        Source = PowerBI.Dataflows(null),
        #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e" = Source{[workspaceId = "d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"]}[Data],
        #"54e82e7b-6c12-4f7d-a631-69cd553c9d59" = #"d3bcff4e-0826-4d4d-9dc6-bbbc58be488e"{[dataflowId = "54e82e7b-6c12-4f7d-a631-69cd553c9d59"]}[Data],
        PHSM1 = #"54e82e7b-6c12-4f7d-a631-69cd553c9d59"{[entity = "PHSM"]}[Data]
    in
        PHSM1;