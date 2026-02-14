import pandas as pd


def main() -> None:
    input_path = "TEST_SET_X.csv"
    output_path = "TEST_SET_X_fixed.csv"

    df = pd.read_csv(input_path)

    date_cols = ["last_scraped", "host_since", "first_review", "last_review"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
