# from IPython.display import Markdown, display
# import pandas as pd
# import numpy as np


# def extract_treatment_effects(model_results):
#     """Extract treatment effects from all fitted models."""
#     results = []

#     for outcome, methods in model_results.items():
#         for method, model_data in methods.items():
#             if model_data is None:  # Model failed to converge
#                 continue

#             if 'logistic_regression' in model_data:
#                 model = model_data['logistic_regression'].model
#                 n_obs = int(model.nobs)

#                 # Count events for binary outcomes
#                 y = model_data['logistic_regression'].y
#                 n_events = int(y.sum())

#                 # Find INPWT coefficient
#                 inpwt_params = [p for p in model.params.index if 'INPWT' in p]
#                 if inpwt_params:
#                     param_name = inpwt_params[0]
#                     coef = model.params[param_name]
#                     pval = model.pvalues[param_name]
#                     ci_lower, ci_upper = model.conf_int().loc[param_name]

#                     results.append({
#                         'Outcome': outcome,
#                         'Method': method.replace('_', ' ').title(),
#                         'n': n_obs,
#                         'Events': n_events,
#                         'Coefficient': round(coef, 3),
#                         'OR': round(np.exp(coef), 2),
#                         '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
#                         'p-value': f"{pval:.3f}"
#                     })

#             elif 'negative_binomial' in model_data:
#                 model = model_data['negative_binomial'].model
#                 n_obs = int(model.nobs)

#                 # Find INPWT coefficient
#                 inpwt_params = [p for p in model.params.index if 'INPWT' in p]
#                 if inpwt_params:
#                     param_name = inpwt_params[0]
#                     coef = model.params[param_name]
#                     pval = model.pvalues[param_name]
#                     ci_lower, ci_upper = model.conf_int().loc[param_name]

#                     results.append({
#                         'Outcome': outcome,
#                         'Method': method.replace('_', ' ').title(),
#                         'n': n_obs,
#                         'Coefficient': round(coef, 3),
#                         'IRR': round(np.exp(coef), 2),
#                         '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
#                         'p-value': f"{pval:.3f}"
#                     })

#     return pd.DataFrame(results)


# def create_selection_summary_table(model_results, response_variables):
#     """Table 1: Variable Selection Summary by Outcome and Method"""
#     results = []

#     for outcome in response_variables:
#         if outcome not in model_results:
#             continue

#         for method in ['univariate', 'elastic_net', 'consensus']:
#             if method in model_results[outcome] and model_results[outcome][method] is not None:
#                 model_data = model_results[outcome][method]
#                 # Subtract 1 for treatment variable to get covariate count
#                 n_covariates = len(model_data['selected_variables']) - 1

#                 results.append({
#                     'Outcome': outcome,
#                     'Method': method.replace('_', ' ').title(),
#                     'Covariates Selected': n_covariates,
#                     'Model Converged': '✓',
#                     'Notes': '—'
#                 })
#             else:
#                 # Model was attempted but failed to converge
#                 results.append({
#                     'Outcome': outcome,
#                     'Method': method.replace('_', ' ').title(),
#                     'Covariates Selected': '—',
#                     'Model Converged': '✗',
#                     'Notes': 'Failed to converge'
#                 })

#     return pd.DataFrame(results)


# def create_variable_selection_table(model_results, outcome):
#     """Table showing which variables were selected by each method for a specific outcome."""
#     if outcome not in model_results:
#         return pd.DataFrame()

#     all_vars = set()
#     method_vars = {}

#     for method, model_data in model_results[outcome].items():
#         if model_data is not None:
#             vars_selected = model_data['selected_variables']
#             method_vars[method] = set(vars_selected)
#             all_vars.update(vars_selected)

#     if len(all_vars) == 0:
#         return pd.DataFrame()

#     # Sort: INPWT first, then alphabetical
#     sorted_vars = sorted(all_vars, key=lambda x: (x != 'INPWT', x))

#     results = []
#     for var in sorted_vars:
#         row = {'Variable': var}
#         for method in ['univariate', 'elastic_net', 'consensus']:
#             if method in method_vars:
#                 row[method.replace('_', ' ').title()] = '✓' if var in method_vars[method] else ''
#             else:
#                 row[method.replace('_', ' ').title()] = '—'
#         results.append(row)

#     return pd.DataFrame(results)


# def extract_treatment_effects_by_outcome(model_results, outcome):
#     """Extract treatment effects for a specific outcome across methods."""
#     results = []

#     if outcome not in model_results:
#         return pd.DataFrame()

#     for method, model_data in model_results[outcome].items():
#         if model_data is None:
#             continue

#         if 'logistic_regression' in model_data:
#             model = model_data['logistic_regression'].model
#             n_obs = int(model.nobs)
#             y = model_data['logistic_regression'].y
#             n_events = int(y.sum())

#             inpwt_params = [p for p in model.params.index if 'INPWT' in p]
#             if inpwt_params:
#                 param_name = inpwt_params[0]
#                 coef = model.params[param_name]
#                 pval = model.pvalues[param_name]
#                 ci_lower, ci_upper = model.conf_int().loc[param_name]

#                 results.append({
#                     'Model': method.replace('_', ' ').title(),
#                     'n': n_obs,
#                     'Events': n_events,
#                     'Coefficient': round(coef, 3),
#                     'OR': round(np.exp(coef), 2),
#                     '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
#                     'p-value': f"{pval:.3f}"
#                 })

#         elif 'negative_binomial' in model_data:
#             model = model_data['negative_binomial'].model
#             n_obs = int(model.nobs)

#             inpwt_params = [p for p in model.params.index if 'INPWT' in p]
#             if inpwt_params:
#                 param_name = inpwt_params[0]
#                 coef = model.params[param_name]
#                 pval = model.pvalues[param_name]
#                 ci_lower, ci_upper = model.conf_int().loc[param_name]

#                 results.append({
#                     'Model': method.replace('_', ' ').title(),
#                     'n': n_obs,
#                     'Coefficient': round(coef, 3),
#                     'IRR': round(np.exp(coef), 2),
#                     '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
#                     'p-value': f"{pval:.3f}"
#                 })

#     return pd.DataFrame(results)


# def extract_covariate_effects(model_results, outcome, method='univariate', top_n=None):
#     """Extract covariate effects from a specific model."""
#     if outcome not in model_results or method not in model_results[outcome]:
#         return pd.DataFrame()

#     model_data = model_results[outcome][method]
#     if model_data is None:
#         return pd.DataFrame()

#     if 'logistic_regression' in model_data:
#         model = model_data['logistic_regression'].model
#         effect_type = 'OR'
#     elif 'negative_binomial' in model_data:
#         model = model_data['negative_binomial'].model
#         effect_type = 'IRR'
#     else:
#         return pd.DataFrame()

#     results = []
#     for param_name in model.params.index:
#         if param_name == 'const' or param_name == 'alpha' or 'INPWT' in param_name:
#             continue

#         coef = model.params[param_name]
#         pval = model.pvalues[param_name]
#         ci_lower, ci_upper = model.conf_int().loc[param_name]

#         results.append({
#             'Predictor': param_name,
#             'Coefficient': round(coef, 3),
#             effect_type: round(np.exp(coef), 2),
#             '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
#             'p-value': f"{pval:.3f}"
#         })

#     df = pd.DataFrame(results)

#     # Sort by p-value and optionally take top N
#     if len(df) > 0:
#         df['p_numeric'] = df['p-value'].astype(float)
#         df = df.sort_values('p_numeric')
#         df = df.drop('p_numeric', axis=1)

#         if top_n:
#             df = df.head(top_n)

#     return df


# def df_to_markdown(df):
#     """Convert dataframe to markdown table."""
#     if len(df) == 0:
#         return "No data available"

#     # Header
#     header = "| " + " | ".join(str(col) for col in df.columns) + " |"
#     separator = "|" + "|".join([" --- " for _ in df.columns]) + "|"

#     # Rows
#     rows = []
#     for _, row in df.iterrows():
#         rows.append("| " + " | ".join(str(x) for x in row) + " |")

#     return "\n".join([header, separator] + rows)


# # ============================================================================
# # GENERATE RESULTS SECTION
# # ============================================================================

# def generate_results_section(model_results, response_variables):
#     """
#     Generate complete results section with all tables auto-updating from model_results.

#     This version accounts for the new analysis workflow:
#     - EPV-based covariate limiting (10 events per variable for binary, 15 obs per variable for count)
#     - Top-N selection when elastic net exceeds limits
#     - Convergence failures reported explicitly
#     - Three selection methods: univariate, elastic net, consensus
#     """

#     # Generate all tables
#     table1 = create_selection_summary_table(model_results, response_variables)
#     table1_md = df_to_markdown(table1)

#     # Check which outcomes have converged models to report on
#     outcomes_to_report = {}
#     for outcome in response_variables:
#         if outcome in model_results:
#             converged_methods = [m for m, d in model_results[outcome].items() if d is not None]
#             if converged_methods:
#                 outcomes_to_report[outcome] = converged_methods

#     results_markdown = f"""
# ## Results

# ### Overview of Variable Selection and Modeling

# We employed a systematic two-stage approach to variable selection and model fitting. First, we applied two independent variable selection methods—univariate testing (p < 0.1) and elastic net regularization (L1 ratio = 0.5, 5-fold cross-validation)—to identify candidate predictors for each outcome. Second, we determined the maximum number of covariates that could be reliably estimated given our sample size, using established guidelines: 10 events per variable for binary outcomes (Peduzzi et al., 1996) and 15 observations per variable for count outcomes, the latter being more conservative to account for increased variance in count data.

# When either selection method identified more variables than the sample size threshold permitted, we retained only the top N variables ranked by statistical strength (smallest p-values for univariate, largest absolute coefficients for elastic net). The treatment variable (incisional negative pressure wound therapy, INPWT) was included in all models regardless of selection and counted toward the variable limit.

# We attempted to fit three models per outcome: (1) univariate selection, (2) elastic net selection, and (3) consensus (variables selected by both methods). For outcomes where models failed to converge despite appropriate sample size considerations, we concluded the data were insufficient to support that covariate structure and report this limitation explicitly.

# **Table 1. Variable Selection and Model Convergence by Outcome and Method**

# {table1_md}

# """

#     # Add outcome-specific sections

#     # ========================================================================
#     # SURGICAL LENGTH OF STAY
#     # ========================================================================
#     if 'Surgical Length of Stay' in outcomes_to_report:
#         var_table = create_variable_selection_table(model_results, 'Surgical Length of Stay')
#         var_table_md = df_to_markdown(var_table)

#         treatment_table = extract_treatment_effects_by_outcome(model_results, 'Surgical Length of Stay')
#         treatment_table_md = df_to_markdown(treatment_table)

#         covariate_table = extract_covariate_effects(model_results, 'Surgical Length of Stay', 'univariate', top_n=10)
#         covariate_table_md = df_to_markdown(covariate_table)

#         results_markdown += f"""
# ### Surgical Length of Stay

# The univariate method (p < 0.1) selected 6 covariates, all within the sample size limit of 9 covariates (152 observations / 15 = 10 total variables, minus 1 for treatment). Elastic net selected 13 covariates; we retained the top 9 by coefficient magnitude. However, 6 of these 9 variables exhibited perfect separation (certain predictor levels had 100% of observations at specific outcome values), and the model failed to converge. The consensus approach was not applicable as there was no overlap between univariate and elastic net selections after EPV limiting. Thus, only the univariate model converged successfully.

# **Variables in Univariate Model**

# {var_table_md}

# Two variables exhibited perfect separation (Hospital Discharge Destination: Acute Care Hospital; Duration of Surgical Procedure: 29 minutes), yet the univariate model converged successfully. This suggests these patterns, while extreme, did not prevent parameter estimation in this specific covariate configuration.

# **Treatment Effect on Surgical Length of Stay**

# {treatment_table_md}

# INPWT was associated with a 9% reduction in expected length of stay (IRR 0.91, 95% CI 0.78–1.05), though this did not reach statistical significance (p = 0.190).

# **Key Covariate Effects (Univariate Model)**

# {covariate_table_md}

# Discharge destination was the strongest predictor of length of stay. Patients discharged to other facilities (not home or skilled care) had 42% longer stays (IRR 1.35, 95% CI 1.14–1.75, p = 0.001), and those discharged to skilled care facilities had 33% longer stays (IRR 1.33, 95% CI 1.06–1.67, p = 0.015), compared to discharge home.

# Diabetes status also significantly predicted length of stay. Patients without diabetes had 27% shorter stays compared to insulin-dependent diabetics (IRR 0.73, 95% CI 0.60–0.87, p < 0.001), and those with non-insulin-dependent diabetes had 27% shorter stays (IRR 0.73, 95% CI 0.58–0.90, p = 0.004).

# """

#     # ========================================================================
#     # POSTOPERATIVE SEPSIS
#     # ========================================================================
#     if 'Postop Sepsis Occurrence' in outcomes_to_report:
#         treatment_table = extract_treatment_effects_by_outcome(model_results, 'Postop Sepsis Occurrence')
#         treatment_table_md = df_to_markdown(treatment_table)

#         results_markdown += f"""
# ### Postoperative Sepsis

# With only 27 sepsis events, the events-per-variable guideline (10 events per variable) permitted a maximum of 2 total variables (treatment + 1 covariate). Neither univariate testing (p < 0.1) nor elastic net regularization identified any covariates. All three models (univariate, elastic net, and consensus) were therefore identical, consisting of the treatment variable only.

# **Treatment Effect on Postoperative Sepsis (All Models)**

# {treatment_table_md}

# INPWT demonstrated a significant protective association with postoperative sepsis. Patients receiving INPWT had 64% lower odds of developing sepsis (OR 0.36, 95% CI 0.16–0.85, p = 0.019). This effect remained robust across all modeling approaches. The absence of selected covariates suggests that, within this dataset and given the statistical power constraints, the treatment effect on sepsis was not confounded by the measured patient or procedural characteristics that were available for analysis.

# """

#     # ========================================================================
#     # POSTOPERATIVE SSI
#     # ========================================================================
#     if 'Postop SSI Occurrence' in model_results:
#         results_markdown += """
# ### Postoperative Surgical Site Infection

# With only 26 SSI events, the events-per-variable guideline permitted a maximum of 2 total variables (treatment + 1 covariate). Both univariate testing and elastic net selection identified the same covariate: number of postoperative organ/space SSIs at time of surgery. However, this variable exhibited perfect separation—all patients with organ/space SSI at surgery also had subsequent SSI. This perfect collinearity prevented model convergence for both univariate and elastic net approaches.

# The elastic net initially selected 14 variables but was limited to the top 1 by coefficient magnitude, which happened to be the same separated variable. No multivariable models could be fit for this outcome due to the separation issue combined with insufficient events for including additional predictors.

# """

#     # ========================================================================
#     # READMISSION
#     # ========================================================================
#     if 'Readmission likely related to Primary Procedure' in outcomes_to_report:
#         treatment_table = extract_treatment_effects_by_outcome(model_results, 'Readmission likely related to Primary Procedure')
#         treatment_table_md = df_to_markdown(treatment_table)

#         results_markdown += f"""
# ### Readmission Related to Primary Procedure

# With only 23 readmission events, the sample size permitted a maximum of 2 total variables (treatment + 1 covariate). Univariate testing identified 8 potentially associated covariates but was limited to the top 1 by p-value: number of postoperative organ/space SSIs. This variable exhibited perfect separation, causing the univariate model to fail to converge.

# Elastic net selected no covariates, allowing a treatment-only model to be fit successfully.

# **Treatment Effect on Readmission (Elastic Net Model)**

# {treatment_table_md}

# No evidence of a treatment effect on readmission was observed (OR 0.98, 95% CI 0.40–2.41, p = 0.959). The pseudo R-squared was near zero (0.00002), indicating the treatment explained essentially none of the variation in readmission risk. This null finding should be interpreted cautiously given the small number of events and lack of covariate adjustment.

# """

#     results_markdown += """
# ### Summary

# INPWT showed a significant protective association with postoperative sepsis, with a 64% reduction in odds (OR 0.36, 95% CI 0.16–0.85, p = 0.019). This effect was not confounded by measured patient characteristics that could be included given sample size constraints. For surgical length of stay, INPWT demonstrated a modest 9% reduction (IRR 0.91) that did not reach statistical significance (p = 0.190).

# Discharge disposition and diabetes status were the strongest determinants of length of stay in the univariate model. Discharge to non-home settings was associated with 33–42% longer stays, and absence of insulin-dependent diabetes was associated with 27% shorter stays compared to insulin-dependent diabetes.

# Sample size limitations precluded definitive analysis of SSI (26 events) and readmission (23 events) outcomes. The small event counts relative to the number of candidate predictors resulted in models that either could not be fit due to separation issues (SSI, readmission univariate) or had insufficient power for covariate adjustment (readmission elastic net, treatment-only model). The elastic net selection for surgical length of stay identified 13 variables, but including the top 9 by coefficient magnitude resulted in convergence failure due to separation in 6 variables, highlighting the challenge of multivariable modeling with moderate sample sizes and complex covariate patterns.

# These findings underscore the importance of sample size considerations in surgical outcomes research. Future studies of these endpoints may require larger samples or alternative analytical approaches such as penalized regression methods specifically designed to handle separation, propensity score methods to reduce dimensionality, or Bayesian approaches that can incorporate prior information to stabilize estimates with sparse data.
# """

#     return results_markdown


from IPython.display import Markdown, display
import pandas as pd
import numpy as np


def extract_treatment_effects(model_results):
    """Extract treatment effects from all fitted models."""
    results = []

    for outcome, methods in model_results.items():
        for method, model_data in methods.items():
            if model_data is None:  # Model failed to converge
                continue

            if "logistic_regression" in model_data:
                model = model_data["logistic_regression"].model
                n_obs = int(model.nobs)

                # Count events for binary outcomes
                y = model_data["logistic_regression"].y
                n_events = int(y.sum())

                # Find INPWT coefficient
                inpwt_params = [p for p in model.params.index if "INPWT" in p]
                if inpwt_params:
                    param_name = inpwt_params[0]
                    coef = model.params[param_name]
                    pval = model.pvalues[param_name]
                    ci_lower, ci_upper = model.conf_int().loc[param_name]

                    results.append(
                        {
                            "Outcome": outcome,
                            "Method": method.replace("_", " ").title(),
                            "n": n_obs,
                            "Events": n_events,
                            "Coefficient": round(coef, 3),
                            "OR": round(np.exp(coef), 2),
                            "95% CI": f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                            "p-value": f"{pval:.3f}",
                        }
                    )

            elif "negative_binomial" in model_data:
                model = model_data["negative_binomial"].model
                n_obs = int(model.nobs)

                # Find INPWT coefficient
                inpwt_params = [p for p in model.params.index if "INPWT" in p]
                if inpwt_params:
                    param_name = inpwt_params[0]
                    coef = model.params[param_name]
                    pval = model.pvalues[param_name]
                    ci_lower, ci_upper = model.conf_int().loc[param_name]

                    results.append(
                        {
                            "Outcome": outcome,
                            "Method": method.replace("_", " ").title(),
                            "n": n_obs,
                            "Coefficient": round(coef, 3),
                            "IRR": round(np.exp(coef), 2),
                            "95% CI": f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                            "p-value": f"{pval:.3f}",
                        }
                    )

    return pd.DataFrame(results)


def create_selection_summary_table(model_results, response_variables):
    """Table 1: Variable Selection Summary by Outcome and Method"""
    results = []

    for outcome in response_variables:
        if outcome not in model_results:
            continue

        for method in ["unadjusted", "univariate", "elastic_net", "consensus"]:
            if (
                method in model_results[outcome]
                and model_results[outcome][method] is not None
            ):
                model_data = model_results[outcome][method]

                if method == "unadjusted":
                    n_covariates = 0  # Treatment only
                else:
                    # Subtract 1 for treatment variable to get covariate count
                    n_covariates = len(model_data["selected_variables"]) - 1

                results.append(
                    {
                        "Outcome": outcome,
                        "Method": method.replace("_", " ").title(),
                        "Covariates Selected": n_covariates,
                        "Model Converged": "✓",
                        "Notes": "—",
                    }
                )
            else:
                # Model was attempted but failed to converge, or wasn't attempted
                # Don't report missing consensus models if univariate or elastic net failed
                if method == "consensus":
                    # Only report consensus if both univariate and elastic net exist
                    has_univariate = (
                        "univariate" in model_results[outcome]
                        and model_results[outcome]["univariate"] is not None
                    )
                    has_elastic = (
                        "elastic_net" in model_results[outcome]
                        and model_results[outcome]["elastic_net"] is not None
                    )
                    if not (has_univariate and has_elastic):
                        continue  # Skip consensus row if either parent method failed

                results.append(
                    {
                        "Outcome": outcome,
                        "Method": method.replace("_", " ").title(),
                        "Covariates Selected": "—",
                        "Model Converged": "✗",
                        "Notes": "Failed to converge",
                    }
                )

    return pd.DataFrame(results)


def create_variable_selection_table(model_results, outcome):
    """Table showing which variables were selected by each method for a specific outcome."""
    if outcome not in model_results:
        return pd.DataFrame()

    all_vars = set()
    method_vars = {}

    for method, model_data in model_results[outcome].items():
        if model_data is not None:
            vars_selected = model_data["selected_variables"]
            method_vars[method] = set(vars_selected)
            all_vars.update(vars_selected)

    if len(all_vars) == 0:
        return pd.DataFrame()

    # Sort: INPWT first, then alphabetical
    sorted_vars = sorted(all_vars, key=lambda x: (x != "INPWT", x))

    results = []
    for var in sorted_vars:
        row = {"Variable": var}
        for method in ["univariate", "elastic_net", "consensus"]:
            if method in method_vars:
                row[method.replace("_", " ").title()] = (
                    "✓" if var in method_vars[method] else ""
                )
            else:
                row[method.replace("_", " ").title()] = "—"
        results.append(row)

    return pd.DataFrame(results)


def extract_treatment_effects_by_outcome(model_results, outcome):
    """Extract treatment effects for a specific outcome across methods."""
    results = []

    if outcome not in model_results:
        return pd.DataFrame()

    for method, model_data in model_results[outcome].items():
        if model_data is None:
            continue

        if "logistic_regression" in model_data:
            model = model_data["logistic_regression"].model
            n_obs = int(model.nobs)
            y = model_data["logistic_regression"].y
            n_events = int(y.sum())

            inpwt_params = [p for p in model.params.index if "INPWT" in p]
            if inpwt_params:
                param_name = inpwt_params[0]
                coef = model.params[param_name]
                pval = model.pvalues[param_name]
                ci_lower, ci_upper = model.conf_int().loc[param_name]

                results.append(
                    {
                        "Model": method.replace("_", " ").title(),
                        "n": n_obs,
                        "Events": n_events,
                        "Coefficient": round(coef, 3),
                        "OR": round(np.exp(coef), 2),
                        "95% CI": f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                        "p-value": f"{pval:.3f}",
                    }
                )

        elif "negative_binomial" in model_data:
            model = model_data["negative_binomial"].model
            n_obs = int(model.nobs)

            inpwt_params = [p for p in model.params.index if "INPWT" in p]
            if inpwt_params:
                param_name = inpwt_params[0]
                coef = model.params[param_name]
                pval = model.pvalues[param_name]
                ci_lower, ci_upper = model.conf_int().loc[param_name]

                results.append(
                    {
                        "Model": method.replace("_", " ").title(),
                        "n": n_obs,
                        "Coefficient": round(coef, 3),
                        "IRR": round(np.exp(coef), 2),
                        "95% CI": f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                        "p-value": f"{pval:.3f}",
                    }
                )

    return pd.DataFrame(results)


def extract_covariate_effects(model_results, outcome, method="univariate", top_n=None):
    """Extract covariate effects from a specific model."""
    if outcome not in model_results or method not in model_results[outcome]:
        return pd.DataFrame()

    model_data = model_results[outcome][method]
    if model_data is None:
        return pd.DataFrame()

    if "logistic_regression" in model_data:
        model = model_data["logistic_regression"].model
        effect_type = "OR"
    elif "negative_binomial" in model_data:
        model = model_data["negative_binomial"].model
        effect_type = "IRR"
    else:
        return pd.DataFrame()

    results = []
    for param_name in model.params.index:
        if param_name == "const" or param_name == "alpha" or "INPWT" in param_name:
            continue

        coef = model.params[param_name]
        pval = model.pvalues[param_name]
        ci_lower, ci_upper = model.conf_int().loc[param_name]

        results.append(
            {
                "Predictor": param_name,
                "Coefficient": round(coef, 3),
                effect_type: round(np.exp(coef), 2),
                "95% CI": f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                "p-value": f"{pval:.3f}",
            }
        )

    df = pd.DataFrame(results)

    # Sort by p-value and optionally take top N
    if len(df) > 0:
        df["p_numeric"] = df["p-value"].astype(float)
        df = df.sort_values("p_numeric")
        df = df.drop("p_numeric", axis=1)

        if top_n:
            df = df.head(top_n)

    return df


def df_to_markdown(df):
    """Convert dataframe to markdown table."""
    if len(df) == 0:
        return "No data available"

    # Header
    header = "| " + " | ".join(str(col) for col in df.columns) + " |"
    separator = "|" + "|".join([" --- " for _ in df.columns]) + "|"

    # Rows
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(x) for x in row) + " |")

    return "\n".join([header, separator] + rows)


# ============================================================================
# GENERATE RESULTS SECTION
# ============================================================================


def generate_results_section(model_results, response_variables):
    """
    Generate complete results section with all tables auto-updating from model_results.

    This version accounts for the new analysis workflow:
    - EPV-based covariate limiting (10 events per variable for binary, 15 obs per variable for count)
    - Top-N selection when elastic net exceeds limits
    - Convergence failures reported explicitly
    - Three selection methods: univariate, elastic net, consensus
    """

    # Generate all tables
    table1 = create_selection_summary_table(model_results, response_variables)
    table1_md = df_to_markdown(table1)

    # Check which outcomes have ANY converged models to report on (including unadjusted)
    outcomes_to_report = {}
    for outcome in response_variables:
        if outcome in model_results:
            converged_methods = [
                m for m, d in model_results[outcome].items() if d is not None
            ]
            if converged_methods:
                outcomes_to_report[outcome] = converged_methods

    results_markdown = f"""
## Results

### Overview of Variable Selection and Modeling

We employed a systematic two-stage approach to variable selection and model fitting. First, we applied two independent variable selection methods—univariate testing (p < 0.1) and elastic net regularization (L1 ratio = 0.5, 5-fold cross-validation)—to identify candidate predictors for each outcome. Second, we determined the maximum number of covariates that could be reliably estimated given our sample size, using established guidelines: 10 events per variable for binary outcomes (Peduzzi et al., 1996) and 15 observations per variable for count outcomes, the latter being more conservative to account for increased variance in count data.

When either selection method identified more variables than the sample size threshold permitted, we retained only the top N variables ranked by statistical strength (smallest p-values for univariate, largest absolute coefficients for elastic net). The treatment variable (incisional negative pressure wound therapy, INPWT) was included in all models regardless of selection and counted toward the variable limit.

We attempted to fit four models per outcome to assess treatment effects and confounding: (1) unadjusted (treatment only), (2) univariate selection (treatment + univariate covariates), (3) elastic net selection (treatment + elastic net covariates), and (4) consensus (treatment + variables selected by both methods). Comparison of unadjusted and adjusted estimates allowed assessment of confounding. When adjusted models failed to converge despite appropriate sample size considerations, we reported unadjusted estimates with appropriate caveats about unmeasured or unmeasurable confounding.

**Table 1. Variable Selection and Model Convergence by Outcome and Method**

{table1_md}

"""

    # Add outcome-specific sections

    # ========================================================================
    # SURGICAL LENGTH OF STAY
    # ========================================================================
    if "Surgical Length of Stay" in outcomes_to_report:
        var_table = create_variable_selection_table(
            model_results, "Surgical Length of Stay"
        )
        var_table_md = df_to_markdown(var_table)

        treatment_table = extract_treatment_effects_by_outcome(
            model_results, "Surgical Length of Stay"
        )
        treatment_table_md = df_to_markdown(treatment_table)

        covariate_table = extract_covariate_effects(
            model_results, "Surgical Length of Stay", "univariate", top_n=10
        )
        covariate_table_md = df_to_markdown(covariate_table)

        # Check if unadjusted exists
        has_unadjusted = (
            "unadjusted" in model_results["Surgical Length of Stay"]
            and model_results["Surgical Length of Stay"]["unadjusted"] is not None
        )

        # Extract treatment effect values
        if len(treatment_table) > 0:
            los_row = treatment_table.iloc[0]
            irr_val = los_row["IRR"]
            ci_val = los_row["95% CI"]
            p_val = los_row["p-value"]
            pct_reduction = int((1 - irr_val) * 100)

        results_markdown += f"""
### Surgical Length of Stay

"""

        if has_unadjusted:
            results_markdown += f"""The unadjusted analysis included all 163 patients with complete outcome data. The univariate method (p < 0.1) selected 6 covariates, all within the sample size limit of 9 covariates (152 observations / 15 = 10 total variables, minus 1 for treatment). This model included 130 patients after excluding those with missing covariate data. Elastic net selected 13 covariates; we retained the top 9 by coefficient magnitude. However, 6 of these 9 variables exhibited perfect separation (certain predictor levels had 100% of observations at specific outcome values), and the model failed to converge. The consensus approach was not applicable as there was no overlap between univariate and elastic net selections after EPV limiting.

**Variables Across Models**

{var_table_md}

Two variables in the univariate model exhibited perfect separation (Hospital Discharge Destination: Acute Care Hospital; Duration of Surgical Procedure: 29 minutes), yet the model converged successfully. This suggests these patterns, while extreme, did not prevent parameter estimation in this specific covariate configuration.

**Treatment Effect on Surgical Length of Stay**

{treatment_table_md}

"""
            if len(treatment_table) > 0:
                results_markdown += f"""In the adjusted univariate model, INPWT was associated with a {pct_reduction}% reduction in expected length of stay (IRR {irr_val}, 95% CI {ci_val}, p = {p_val}). """
                if float(p_val) >= 0.05:
                    results_markdown += f"""This did not reach statistical significance.

"""
                else:
                    results_markdown += f"""This was statistically significant.

"""
        else:
            results_markdown += f"""The unadjusted analysis (treatment only) failed to converge, likely due to extreme overdispersion in the full dataset. The univariate method (p < 0.1) selected 6 covariates, all within the sample size limit of 9 covariates (152 observations / 15 = 10 total variables, minus 1 for treatment). This model included 130 patients after excluding those with missing covariate data and converged successfully. Elastic net selected 13 covariates; we retained the top 9 by coefficient magnitude. However, 6 of these 9 variables exhibited perfect separation, and the model failed to converge.

**Variables in Univariate Model**

{var_table_md}

Two variables exhibited perfect separation (Hospital Discharge Destination: Acute Care Hospital; Duration of Surgical Procedure: 29 minutes), yet the univariate model converged successfully. This suggests these patterns, while extreme, did not prevent parameter estimation in this specific covariate configuration.

**Treatment Effect on Surgical Length of Stay (Univariate Model)**

{treatment_table_md}

"""
            if len(treatment_table) > 0:
                results_markdown += f"""In the adjusted univariate model, INPWT was associated with a {pct_reduction}% reduction in expected length of stay (IRR {irr_val}, 95% CI {ci_val}), though this did not reach statistical significance (p = {p_val}). The failure of the unadjusted model to converge prevented direct assessment of confounding, though the adjusted estimate suggests a modest protective effect after accounting for discharge destination and diabetes status.

"""

        results_markdown += f"""**Key Covariate Effects (Univariate Model)**

{covariate_table_md}

Discharge destination was the strongest predictor of length of stay. Patients discharged to other facilities (not home or skilled care) had 42% longer stays (IRR 1.35, 95% CI 1.14-1.75, p = 0.001), and those discharged to skilled care facilities had 33% longer stays (IRR 1.33, 95% CI 1.06-1.67, p = 0.015), compared to discharge home. 

Diabetes status also significantly predicted length of stay. Patients without diabetes had 27% shorter stays compared to insulin-dependent diabetics (IRR 0.73, 95% CI 0.60-0.87, p < 0.001), and those with non-insulin-dependent diabetes had 27% shorter stays (IRR 0.73, 95% CI 0.58-0.90, p = 0.004).

"""

    # ========================================================================
    # POSTOPERATIVE SEPSIS
    # ========================================================================
    if "Postop Sepsis Occurrence" in outcomes_to_report:
        treatment_table = extract_treatment_effects_by_outcome(
            model_results, "Postop Sepsis Occurrence"
        )
        treatment_table_md = df_to_markdown(treatment_table)

        # Extract actual values from table
        sepsis_row = treatment_table.iloc[0]
        or_val = sepsis_row["OR"]
        ci_val = sepsis_row["95% CI"]
        p_val = sepsis_row["p-value"]
        pct_reduction = int((1 - or_val) * 100)

        results_markdown += f"""
### Postoperative Sepsis

With only 27 sepsis events, the events-per-variable guideline (10 events per variable) permitted a maximum of 2 total variables (treatment + 1 covariate). Neither univariate testing (p < 0.1) nor elastic net regularization identified any covariates. All three models (univariate, elastic net, and consensus) were therefore identical, consisting of the treatment variable only.

**Treatment Effect on Postoperative Sepsis (All Models)**

{treatment_table_md}

INPWT demonstrated a significant protective association with postoperative sepsis. Patients receiving INPWT had {pct_reduction}% lower odds of developing sepsis (OR {or_val}, 95% CI {ci_val}, p = {p_val}). This effect remained robust across all modeling approaches. The absence of selected covariates suggests that, within this dataset and given the statistical power constraints, the treatment effect on sepsis was not confounded by the measured patient or procedural characteristics that were available for analysis.

"""

    # ========================================================================
    # POSTOPERATIVE SSI
    # ========================================================================
    if "Postop SSI Occurrence" in model_results:
        # Check if unadjusted model exists
        has_unadjusted = (
            "unadjusted" in model_results["Postop SSI Occurrence"]
            and model_results["Postop SSI Occurrence"]["unadjusted"] is not None
        )

        if has_unadjusted:
            treatment_table = extract_treatment_effects_by_outcome(
                model_results, "Postop SSI Occurrence"
            )
            treatment_table_md = df_to_markdown(treatment_table)

            results_markdown += f"""
### Postoperative Surgical Site Infection

With only 26 SSI events, the events-per-variable guideline permitted a maximum of 2 total variables (treatment + 1 covariate). Both univariate testing and elastic net selection identified the same covariate: number of postoperative organ/space SSIs at time of surgery. However, this variable exhibited perfect separation—all patients with organ/space SSI at surgery also had subsequent SSI. This perfect collinearity prevented convergence for both univariate and elastic net approaches.

**Treatment Effect on SSI (Unadjusted Model Only)**

{treatment_table_md}

In unadjusted analysis, INPWT showed no evidence of protective effect against SSI (OR 0.71, 95% CI 0.28-1.81, p = 0.47). Although adjusted models could not be fit due to perfect separation in the selected covariate, the null unadjusted finding suggests adjustment is unlikely to reveal a hidden treatment benefit. The inability to adjust for organ/space SSI at surgery—itself a strong predictor—limits causal interpretation, but the wide confidence interval and null point estimate indicate the data provide little evidence for or against a treatment effect.

"""
        else:
            results_markdown += """
### Postoperative Surgical Site Infection

With only 26 SSI events, the events-per-variable guideline permitted a maximum of 2 total variables (treatment + 1 covariate). Both univariate testing and elastic net selection identified the same covariate: number of postoperative organ/space SSIs at time of surgery. However, this variable exhibited perfect separation—all patients with organ/space SSI at surgery also had subsequent SSI. This perfect collinearity prevented model convergence for both univariate and elastic net approaches. No models could be fit for this outcome due to the separation issue combined with insufficient events for including additional predictors.

"""

    # ========================================================================
    # READMISSION
    # ========================================================================
    if "Readmission likely related to Primary Procedure" in outcomes_to_report:
        treatment_table = extract_treatment_effects_by_outcome(
            model_results, "Readmission likely related to Primary Procedure"
        )
        treatment_table_md = df_to_markdown(treatment_table)

        # Extract values
        readm_row = treatment_table.iloc[0]
        or_val = readm_row["OR"]
        ci_val = readm_row["95% CI"]
        p_val = readm_row["p-value"]

        results_markdown += f"""
### Readmission Related to Primary Procedure

With only 23 readmission events, the sample size permitted a maximum of 2 total variables (treatment + 1 covariate). Univariate testing identified 8 potentially associated covariates but was limited to the top 1 by p-value: number of postoperative organ/space SSIs. This variable exhibited perfect separation, causing the univariate model to fail to converge.

Elastic net selected no covariates, allowing a treatment-only model to be fit successfully.

**Treatment Effect on Readmission**

{treatment_table_md}

No evidence of a treatment effect on readmission was observed (OR {or_val}, 95% CI {ci_val}, p = {p_val}). The pseudo R-squared was near zero (0.00002), indicating the treatment explained essentially none of the variation in readmission risk. This null finding should be interpreted cautiously given the small number of events and lack of covariate adjustment.

"""

    results_markdown += """
### Summary

"""

    # Dynamically extract summary values
    summary_values = {}

    # Sepsis
    if "Postop Sepsis Occurrence" in outcomes_to_report:
        sepsis_table = extract_treatment_effects_by_outcome(
            model_results, "Postop Sepsis Occurrence"
        )
        if len(sepsis_table) > 0:
            sepsis_row = sepsis_table.iloc[0]
            summary_values["sepsis_or"] = sepsis_row["OR"]
            summary_values["sepsis_ci"] = sepsis_row["95% CI"]
            summary_values["sepsis_p"] = sepsis_row["p-value"]
            summary_values["sepsis_pct"] = int((1 - sepsis_row["OR"]) * 100)

    # LOS
    if "Surgical Length of Stay" in outcomes_to_report:
        los_table = extract_treatment_effects_by_outcome(
            model_results, "Surgical Length of Stay"
        )
        if len(los_table) > 0:
            los_row = los_table.iloc[0]
            summary_values["los_irr"] = los_row["IRR"]
            summary_values["los_ci"] = los_row["95% CI"]
            summary_values["los_p"] = los_row["p-value"]
            summary_values["los_pct"] = int((1 - los_row["IRR"]) * 100)

    # SSI
    if "Postop SSI Occurrence" in outcomes_to_report:
        ssi_table = extract_treatment_effects_by_outcome(
            model_results, "Postop SSI Occurrence"
        )
        if len(ssi_table) > 0:
            ssi_row = ssi_table.iloc[0]
            summary_values["ssi_or"] = ssi_row["OR"]
            summary_values["ssi_ci"] = ssi_row["95% CI"]
            summary_values["ssi_p"] = ssi_row["p-value"]

    # Readmission
    if "Readmission likely related to Primary Procedure" in outcomes_to_report:
        readm_table = extract_treatment_effects_by_outcome(
            model_results, "Readmission likely related to Primary Procedure"
        )
        if len(readm_table) > 0:
            readm_row = readm_table.iloc[0]
            summary_values["readm_or"] = readm_row["OR"]
            summary_values["readm_ci"] = readm_row["95% CI"]
            summary_values["readm_p"] = readm_row["p-value"]

    # Build summary with dynamic values
    if "sepsis_or" in summary_values:
        results_markdown += f"""INPWT showed a significant protective association with postoperative sepsis in all models (unadjusted and adjusted were identical as no covariates were selected), with a {summary_values['sepsis_pct']}% reduction in odds (OR {summary_values['sepsis_or']}, 95% CI {summary_values['sepsis_ci']}, p = {summary_values['sepsis_p']}). This effect was robust and was not confounded by measured patient characteristics that could be included given sample size constraints.

"""

    if "los_irr" in summary_values:
        results_markdown += f"""For surgical length of stay, only the adjusted univariate model converged, showing a {summary_values['los_pct']}% reduction in expected stay (IRR {summary_values['los_irr']}, 95% CI {summary_values['los_ci']}, p = {summary_values['los_p']}) that did not reach statistical significance. The unadjusted model failed to converge, likely due to extreme overdispersion in the complete dataset, preventing direct assessment of confounding. The adjusted model identified discharge destination and diabetes status as the strongest predictors of length of stay.

"""

    if "ssi_or" in summary_values and "readm_or" in summary_values:
        results_markdown += f"""For SSI and readmission, unadjusted analyses showed null findings (SSI: OR {summary_values['ssi_or']}, 95% CI {summary_values['ssi_ci']}, p = {summary_values['ssi_p']}; Readmission: OR {summary_values['readm_or']}, 95% CI {summary_values['readm_ci']}, p = {summary_values['readm_p']}). Adjusted models could not be fit due to perfect separation in selected covariates or insufficient events. The null unadjusted estimates with point estimates near 1.0 provide no suggestion that adjustment would reveal substantial treatment benefits.

"""

    results_markdown += """Discharge disposition and diabetes status were the strongest determinants of length of stay in adjusted analysis. Discharge to non-home settings was associated with 33-42% longer stays, and absence of insulin-dependent diabetes was associated with 27% shorter stays compared to insulin-dependent diabetes.

Sample size limitations precluded definitive adjusted analysis of SSI (26 events) and readmission (23 events) outcomes. The small event counts relative to the number of candidate predictors resulted in models that could not be fit due to separation issues. For surgical length of stay, elastic net selection identified 13 variables, but including the top 9 by coefficient magnitude resulted in convergence failure due to separation in 6 variables. Paradoxically, the treatment-only model also failed to converge, highlighting significant overdispersion challenges in count outcome modeling with this dataset.

These findings underscore the importance of sample size considerations in surgical outcomes research and the value of attempting both unadjusted and adjusted analyses. When unadjusted models converge, they provide important context even when adjustment fails. The SSI and readmission results demonstrate that null unadjusted findings, while limited by potential confounding, can still inform clinical decision-making and future research planning. Future studies of these endpoints may require larger samples, alternative analytical approaches such as penalized regression methods specifically designed to handle separation and overdispersion, propensity score methods to reduce dimensionality, or Bayesian approaches that can incorporate prior information to stabilize estimates with sparse data.
"""

    return results_markdown


# Example usage:
# results_md = generate_results_section(model_results, response_variables)
# # display(Markdown(results_md))
# print(results_md)
#
# Or to save:
# with open('results_section.md', 'w') as f:
#     f.write(results_md)
