import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Retail Sales Explorer", layout="wide")
st.title("Retail Sales Explorer")
st.caption("Upload a CSV, explore your data, and predict a numeric value.")

# SECTION 1: Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("This file could not be read. Please upload a valid CSV.")
        st.stop()

    if df.empty:
        st.error("This file has no rows.")
        st.stop()

    st.subheader("1. Preview your data")
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    with st.expander("See raw data", expanded=True):
        st.dataframe(df.head(10), hide_index=True, width="stretch")

    # Identify numeric columns and exclude IDs
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    analysis_cols = [
        col for col in numeric_cols
        if "id" not in col.lower().replace(" ", "")
    ]

    if len(analysis_cols) < 3:
        st.error(
            "Please upload a CSV with at least three numeric columns "
            "that are not ID columns to use the chart and prediction features."
        )
        st.stop()

    # SECTION 2: Choose and create a chart
    st.subheader("2. Explore your data")

    chart_type = st.selectbox(
        "Choose a chart type",
        ["Line chart", "Scatter plot", "Histogram"]
    )

    if chart_type == "Histogram":
        variable = st.selectbox(
            "Column to plot",
            analysis_cols
        )

        fig = px.histogram(
            df,
            x=variable,
            title=f"Distribution of {variable}"
        )

    elif chart_type == "Scatter plot":
        x_col = st.selectbox(
            "X-axis",
            analysis_cols
        )

        y_options = [
            col for col in analysis_cols if col != x_col
        ]

        y_col = st.selectbox(
            "Y-axis",
            y_options
        )

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=f"{y_col} vs. {x_col}"
        )

    else:
        # Use a date column when one is available
        date_cols = [
            col for col in df.columns
            if "date" in col.lower()
        ]

        variable = st.selectbox(
            "Column to plot",
            analysis_cols,
            index=(
                analysis_cols.index("Total Amount")
                if "Total Amount" in analysis_cols else 0
            )
        )

        if date_cols:
            date_col = st.selectbox(
                "Date column",
                date_cols
            )

            line_data = df[[date_col, variable]].copy()
            line_data[date_col] = pd.to_datetime(
                line_data[date_col], errors="coerce"
            )
            line_data = line_data.dropna(subset=[date_col])

            # Sum sales-like values; average other numeric values
            if any(
                word in variable.lower()
                for word in ["amount", "sales", "revenue", "quantity"]
            ):
                line_data = line_data.groupby(
                    date_col, as_index=False
                )[variable].sum()
            else:
                line_data = line_data.groupby(
                    date_col, as_index=False
                )[variable].mean()

            line_data = line_data.sort_values(date_col)

            fig = px.line(
                line_data,
                x=date_col,
                y=variable,
                title=f"{variable} over time",
                markers=True
            )

        else:
            # If no date column exists, plot values by row
            line_data = df[[variable]].reset_index()
            line_data = line_data.rename(columns={"index": "Row"})

            fig = px.line(
                line_data,
                x="Row",
                y=variable,
                title=f"{variable} by row"
            )

    st.plotly_chart(fig, width="stretch")

    # SECTION 3: Linear regression prediction
    st.subheader("3. Predict a numeric value")

    st.write(
        "Choose two numeric inputs and a target. "
        "The model learns from the uploaded data and estimates the target."
    )

    # Set default inputs for the retail dataset
    default_input_1 = (
        "Age" if "Age" in analysis_cols else analysis_cols[0]
    )

    default_input_2 = (
        "Quantity" if "Quantity" in analysis_cols
        else next(c for c in analysis_cols if c != default_input_1)
    )

    target_candidates = [
        c for c in analysis_cols
        if c not in [default_input_1, default_input_2]
    ]

    default_target = (
        "Total Amount" if "Total Amount" in target_candidates
        else target_candidates[0]
    )

    input_1 = st.selectbox(
        "First input",
        analysis_cols,
        index=analysis_cols.index(default_input_1)
    )

    input_2_options = [
        c for c in analysis_cols if c != input_1
    ]

    preferred_input_2 = (
        default_input_2
        if default_input_2 in input_2_options
        else input_2_options[0]
    )

    input_2 = st.selectbox(
        "Second input",
        input_2_options,
        index=input_2_options.index(preferred_input_2)
    )

    target_options = [
        c for c in analysis_cols
        if c not in [input_1, input_2]
    ]

    preferred_target = (
        default_target
        if default_target in target_options
        else target_options[0]
    )

    target = st.selectbox(
        "Target to predict",
        target_options,
        index=target_options.index(preferred_target)
    )

    # Prepare the data used to train the model
    model_data = df[[input_1, input_2, target]].dropna()

    if len(model_data) < 3:
        st.warning("There are not enough complete rows to train the model.")

    else:
        X = model_data[[input_1, input_2]]
        y = model_data[target]

        model = LinearRegression()
        model.fit(X, y)

        st.caption(
            f"Model trained on {len(model_data)} complete rows."
        )

        # Let the user enter new input values
        col1, col2 = st.columns(2)

        with col1:
            value_1 = st.number_input(
                f"Enter {input_1}",
                value=float(model_data[input_1].mean())
            )

        with col2:
            value_2 = st.number_input(
                f"Enter {input_2}",
                value=float(model_data[input_2].mean())
            )

        if st.button("Predict"):
            new_data = pd.DataFrame(
                [[value_1, value_2]],
                columns=[input_1, input_2]
            )

            prediction = model.predict(new_data)[0]

            st.metric(
                label=f"Predicted {target}",
                value=f"{prediction:,.2f}"
            )

    # Required explanation of model limitations
    st.info(
        "About this model: Linear regression estimates a numeric target "
        "using two inputs and assumes a linear relationship. Predictions "
        "may be inaccurate if important variables are missing, the data "
        "is unrepresentative, or the relationship is not linear. "
        "Predictions are estimates, not guarantees."
    )

else:
    st.info(
        "Upload a CSV file to get started. "
        "For all features, the file needs at least three numeric columns "
        "that are not ID columns."
    )
