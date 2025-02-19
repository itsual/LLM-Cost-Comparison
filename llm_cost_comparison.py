import streamlit as st
import pandas as pd

def compute_gemini_cost(
    base_monthly_input_tokens,
    base_monthly_output_tokens,
    token_growth_rate,
    input_token_cost_per_1k,
    output_token_cost_per_1k,
    integration_cost,
    annual_maintenance,
    years=5
):
    """
    Calculates a 5-year cost breakdown for Gemini 2.0 Flash
    given separate input & output token rates.
    """
    rows = []

    monthly_input = base_monthly_input_tokens
    monthly_output = base_monthly_output_tokens

    for year in range(1, years + 1):
        annual_input_tokens = monthly_input * 12
        annual_output_tokens = monthly_output * 12

        # Convert tokens to 1,000-token units
        input_cost = (annual_input_tokens / 1000) * input_token_cost_per_1k
        output_cost = (annual_output_tokens / 1000) * output_token_cost_per_1k

        usage_cost = input_cost + output_cost

        # Labor cost depends on year
        if year == 1:
            labor_cost = integration_cost
        else:
            labor_cost = annual_maintenance

        total_cost = usage_cost + labor_cost

        rows.append({
            'Year': year,
            'Annual Input Tokens': annual_input_tokens,
            'Annual Output Tokens': annual_output_tokens,
            'Usage Cost': usage_cost,
            'Labor/Maint.': labor_cost,
            'Total': total_cost
        })

        # Grow monthly tokens for next year
        monthly_input *= (1 + token_growth_rate / 100.0)
        monthly_output *= (1 + token_growth_rate / 100.0)

    return pd.DataFrame(rows)


def compute_llama_cost(
    gpu_monthly_cost,
    gpu_extra_first_year,
    storage_monthly_cost,
    network_monthly_cost,
    setup_labor,
    ongoing_labor_yearly,
    years=5
):
    """
    Calculates a 5-year cost breakdown for self-hosted LLaMA on GCP.
    """
    rows = []
    
    for year in range(1, years + 1):
        if year == 1:
            gpu_cost = gpu_monthly_cost * 12 + gpu_extra_first_year
            labor_cost = setup_labor + ongoing_labor_yearly
        else:
            gpu_cost = gpu_monthly_cost * 12
            labor_cost = ongoing_labor_yearly

        storage_cost = storage_monthly_cost * 12
        network_cost = network_monthly_cost * 12

        total_cost = gpu_cost + storage_cost + network_cost + labor_cost
        
        rows.append({
            'Year': year,
            'GPU Cost': gpu_cost,
            'Storage Cost': storage_cost,
            'Network Cost': network_cost,
            'Labor Cost': labor_cost,
            'Total': total_cost
        })
    
    return pd.DataFrame(rows)


def main():
    st.title("LLM Cost Comparison Simulator: Gemini 2.0 Flash vs. LLaMA")
    st.markdown("""
    This simulation compares **5-year costs** for:
    - **Gemini 2.0 Flash (API-based)**  
    - **Self-Hosted LLaMA (GCP)**  

    Adjust the sliders in the sidebar to see how usage and infrastructure changes
    affect your total cost.
    """)

    # ------------------------------------------------------
    # Button: Show Assumptions
    # ------------------------------------------------------
    if st.button("Show Assumptions"):
        st.markdown(f"""
        ### Assumptions for Gemini 2.0 Flash
        - **Pricing** from the [official docs](
          https://ai.google.dev/gemini-api/docs/pricing#gemini-2.0-flash):
          - **Input**: \\$0.002 per 1K tokens
          - **Output**: \\$0.004 per 1K tokens
        - **Base Monthly Input/Output Tokens**: You can adjust them in the sidebar.
        - **Annual Growth**: Monthly tokens grow by a fixed percentage each year.
        - **Integration Cost (Year 1)**: Labor to integrate with the API.
        - **Annual Maintenance**: Minor overhead for subsequent years.

        ### Assumptions for LLaMA (Self-Hosted)
        - **GPU Monthly Cost** (e.g., A100 or T4 on GCP).
        - **Extra GPU Cost (Year 1)**: Additional usage for fine-tuning or training.
        - **Storage / Networking**: Monthly cost for persistent disk & egress.
        - **Setup Labor (Year 1)**: Deployment, containerization, config, etc.
        - **Ongoing Labor**: Monitoring, patching, re-tuning, etc.

        ### Time Horizon
        - Both models are compared over **5 years**.

        ### Real-World Adjustments
        - Real-world usage might have higher concurrency or spiky loads.
        - Consider discounts for committed use / reservations on GCP.
        - MLOps labor might be shared among multiple projects.
        - Actual usage patterns may vary significantly month to month.
        - API pricing can differ if usage is spiky or if there's a multi-tier rate.

         ### Qualitative Considerations
        - Data Privacy/Control: Self-hosting provides complete control over data.
        - Customization: Self-hosting allows for model customization.
        - Latency: Potentially lower latency with self-hosting (location dependent).
        - Opportunity Cost: Consider engineering resources for infrastructure management.

        *This example is for educational/demo purposes. Always validate actual pricing with cloud & API providers.*
        """)
    # ------------------------------------------------------

    # Sidebar for Gemini 2.0 Flash
    st.sidebar.header("Gemini 2.0 Flash Parameters")

    base_input = st.sidebar.number_input(
        "Base Monthly Input Tokens",
        min_value=100_000.0,
        max_value=100_000_000.0,
        value=5_000_000.0,
        step=500_000.0
    )

    base_output = st.sidebar.number_input(
        "Base Monthly Output Tokens",
        min_value=100_000.0,
        max_value=100_000_000.0,
        value=3_000_000.0,
        step=500_000.0
    )

    token_growth_rate = st.sidebar.slider(
        "Annual Usage Growth Rate (%)",
        min_value=0, max_value=100, value=10, step=5
    )

    # Default Gemini 2.0 Flash rates (adjust as needed)
    input_token_cost_per_1k = st.sidebar.number_input(
        "Gemini Input Cost per 1K Tokens (USD)",
        min_value=0.0,
        value=0.002,
        step=0.001,
        format="%.3f"
    )

    output_token_cost_per_1k = st.sidebar.number_input(
        "Gemini Output Cost per 1K Tokens (USD)",
        min_value=0.0,
        value=0.004,
        step=0.001,
        format="%.3f"
    )

    integration_cost = st.sidebar.number_input(
        "Integration Cost (Year 1)",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

    annual_maintenance = st.sidebar.number_input(
        "Annual Maintenance (USD, after Year 1)",
        min_value=0.0,
        value=500.0,
        step=100.0
    )

    # Sidebar for LLaMA
    st.sidebar.header("LLaMA (Self-Hosted) Parameters")

    gpu_monthly_cost = st.sidebar.number_input(
        "GPU Monthly Cost (USD)",
        min_value=0.0,
        value=1750.0,
        step=250.0
    )
    
    gpu_extra_first_year = st.sidebar.number_input(
        "Extra GPU (Year 1) for Fine-tuning",
        min_value=0.0,
        value=3500.0,
        step=500.0
    )
    
    storage_monthly_cost = st.sidebar.number_input(
        "Storage Monthly Cost (USD)",
        min_value=0.0,
        value=170.0,
        step=10.0
    )

    network_monthly_cost = st.sidebar.number_input(
        "Networking Monthly Cost (USD)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    setup_labor = st.sidebar.number_input(
        "Setup Labor (Year 1)",
        min_value=0.0,
        value=15000.0,
        step=1000.0
    )

    ongoing_labor_yearly = st.sidebar.number_input(
        "Ongoing Labor (Yearly)",
        min_value=0.0,
        value=7500.0,
        step=1000.0
    )

    # ------------------------------------------------------
    # Compute Costs
    # ------------------------------------------------------
    df_gemini = compute_gemini_cost(
        base_monthly_input_tokens=base_input,
        base_monthly_output_tokens=base_output,
        token_growth_rate=token_growth_rate,
        input_token_cost_per_1k=input_token_cost_per_1k,
        output_token_cost_per_1k=output_token_cost_per_1k,
        integration_cost=integration_cost,
        annual_maintenance=annual_maintenance,
        years=5
    )

    df_llama = compute_llama_cost(
        gpu_monthly_cost=gpu_monthly_cost,
        gpu_extra_first_year=gpu_extra_first_year,
        storage_monthly_cost=storage_monthly_cost,
        network_monthly_cost=network_monthly_cost,
        setup_labor=setup_labor,
        ongoing_labor_yearly=ongoing_labor_yearly,
        years=5
    )

    # ------------------------------------------------------
    # Display: Gemini 2.0 Flash
    # ------------------------------------------------------
    st.subheader("Gemini (2.0 Flash) Yearly Costs")

    gemini_fmt = {
        'Annual Input Tokens': '{:,.0f}',
        'Annual Output Tokens': '{:,.0f}',
        'Usage Cost': '${:,.2f}',
        'Labor/Maint.': '${:,.2f}',
        'Total': '${:,.2f}'
    }
    st.dataframe(df_gemini.style.format(gemini_fmt))

    # ------------------------------------------------------
    # Display: LLaMA
    # ------------------------------------------------------
    st.subheader("LLaMA (Self-Hosted) Yearly Costs")

    llama_fmt = {
        'GPU Cost': '${:,.2f}',
        'Storage Cost': '${:,.2f}',
        'Network Cost': '${:,.2f}',
        'Labor Cost': '${:,.2f}',
        'Total': '${:,.2f}'
    }
    st.dataframe(df_llama.style.format(llama_fmt))

    # ------------------------------------------------------
    # Summaries
    # ------------------------------------------------------
    gemini_total = df_gemini['Total'].sum()
    llama_total = df_llama['Total'].sum()

    st.write(f"**Gemini 5-Year Total**: ${gemini_total:,.2f}")
    st.write(f"**LLaMA 5-Year Total**: ${llama_total:,.2f}")

    diff = llama_total - gemini_total
    if diff > 0:
        st.write(f"LLaMA is **more expensive** by ${diff:,.2f}.")
    elif diff < 0:
        st.write(f"Gemini is **more expensive** by ${abs(diff):,.2f}.")
    else:
        st.write("Both approaches cost exactly the same!")

    # ------------------------------------------------------
    # Cumulative Chart
    # ------------------------------------------------------
    st.subheader("Cumulative 5-Year Cost Chart")
    cum_df = pd.DataFrame({
        "Year": df_gemini['Year'],
        "Gemini Cumulative": df_gemini['Total'].cumsum(),
        "LLaMA Cumulative": df_llama['Total'].cumsum()
    })

    st.line_chart(cum_df.set_index("Year"))


if __name__ == "__main__":
    main()
