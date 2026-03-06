import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected pipeline stages. By default, no stage runs unless flags are provided."
    )
    parser.add_argument("--all", action="store_true", help="Run all stages in order.")
    parser.add_argument("--collect", action="store_true", help="Run data collection.")
    parser.add_argument("--clean", action="store_true", help="Run data cleaning.")
    parser.add_argument("--visualize", action="store_true", help="Run data visualization.")
    parser.add_argument("--analyze", action="store_true", help="Run data analysis.")
    parser.add_argument("--point-forecast", action="store_true", help="Run point forecast model.")
    parser.add_argument("--quantile", action="store_true", help="Run quantile regression model.")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    selected = any(
        [
            args.all,
            args.collect,
            args.clean,
            args.visualize,
            args.analyze,
            args.point_forecast,
            args.quantile,
        ]
    )
    if not selected:
        parser.print_help()
        return

    run_collect = args.all or args.collect
    run_clean = args.all or args.clean
    run_visualize = args.all or args.visualize
    run_analyze = args.all or args.analyze
    run_point = args.all or args.point_forecast
    run_quantile = args.all or args.quantile

    if run_collect:
        import pipelines.Data_collection as Data_collection

        Data_collection.main()
    if run_clean:
        import pipelines.Data_cleaning as Data_cleaning

        Data_cleaning.main()
    if run_visualize:
        import pipelines.Data_visualisation as Data_visualisation

        Data_visualisation.main()
    if run_analyze:
        import pipelines.Data_analysis as Data_analysis

        Data_analysis.main()
    if run_point:
        import src.Point_forecast as Point_forecast

        Point_forecast.main()
    if run_quantile:
        import src.Quantile_regression as Quantile_regression

        Quantile_regression.main()


if __name__ == "__main__":
    main()
