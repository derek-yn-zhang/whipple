# from IPython.display import Markdown, display
# import pandas as pd
# import numpy as np


# def extract_treatment_effects(model_results):
#     """Extract treatment effects from all fitted models."""
#     results = []
    
#     for outcome, methods in model_results.items():
#         for method, model_data in methods.items():
#             if 'logistic_regression' in model_data:
#                 model = model_data['logistic_regression'].model
#                 n_obs = int(model.nobs)
                
#                 # Count events for binary outcomes
#                 y = model_data['logistic_regression'].y
#                 n_events = int(y.sum())
                
#                 # Find INPWT coefficient
#                 inpwt_params = [p for p in model.params.index if 'INPWT_Yes' in p]
#                 if inpwt_params:
#                     param_name = inpwt_params[0]
#                     coef = model.params[param_name]
#                     pval = model.pvalues[param_name]
#                     ci_lower, ci_upper = model.conf_int().loc[param_name]
                    
#                     results.append({
#                         'Outcome': outcome,
#                         'Method': method.title(),
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
#                 inpwt_params = [p for p in model.params.index if 'INPWT_Yes' in p]
#                 if inpwt_params:
#                     param_name = inpwt_params[0]
#                     coef = model.params[param_name]
#                     pval = model.pvalues[param_name]
#                     ci_lower, ci_upper = model.conf_int().loc[param_name]
                    
#                     results.append({
#                         'Outcome': outcome,
#                         'Method': method.title(),
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
#             if method in model_results[outcome]:
#                 model_data = model_results[outcome][method]
#                 n_covariates = len([v for v in model_data['selected_variables'] if v != 'INPWT'])
                
#                 results.append({
#                     'Outcome': outcome,
#                     'Method': method.replace('_', ' ').title(),
#                     'Covariates Selected': n_covariates,
#                     'Model Fit': '✓',
#                     'Reason if Not Fit': '—'
#                 })
#             else:
#                 # Check if it was skipped
#                 results.append({
#                     'Outcome': outcome,
#                     'Method': method.replace('_', ' ').title(),
#                     'Covariates Selected': '—',
#                     'Model Fit': '✗',
#                     'Reason if Not Fit': 'Insufficient events'
#                 })
    
#     return pd.DataFrame(results)


# def create_variable_selection_table(model_results, outcome):
#     """Table showing which variables were selected by each method."""
#     if outcome not in model_results:
#         return pd.DataFrame()
    
#     all_vars = set()
#     method_vars = {}
    
#     for method, model_data in model_results[outcome].items():
#         vars_selected = model_data.get('selected_variables', [])
#         method_vars[method] = set(vars_selected)
#         all_vars.update(vars_selected)
    
#     # Sort: INPWT first, then alphabetical
#     sorted_vars = sorted(all_vars, key=lambda x: (x != 'INPWT', x))
    
#     results = []
#     for var in sorted_vars:
#         row = {'Variable': var}
#         for method in ['univariate', 'elastic_net', 'consensus']:
#             if method in method_vars:
#                 row[method.replace('_', ' ').title()] = '✓' if var in method_vars[method] else ''
#         results.append(row)
    
#     return pd.DataFrame(results)


# def extract_treatment_effects_by_outcome(model_results, outcome):
#     """Extract treatment effects for a specific outcome across methods."""
#     results = []
    
#     if outcome not in model_results:
#         return pd.DataFrame()
    
#     for method, model_data in model_results[outcome].items():
#         if 'logistic_regression' in model_data:
#             model = model_data['logistic_regression'].model
#             n_obs = int(model.nobs)
#             y = model_data['logistic_regression'].y
#             n_events = int(y.sum())
            
#             inpwt_params = [p for p in model.params.index if 'INPWT_Yes' in p]
#             if inpwt_params:
#                 param_name = inpwt_params[0]
#                 coef = model.params[param_name]
#                 pval = model.pvalues[param_name]
#                 ci_lower, ci_upper = model.conf_int().loc[param_name]
                
#                 results.append({
#                     'Model': method.title(),
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
            
#             inpwt_params = [p for p in model.params.index if 'INPWT_Yes' in p]
#             if inpwt_params:
#                 param_name = inpwt_params[0]
#                 coef = model.params[param_name]
#                 pval = model.pvalues[param_name]
#                 ci_lower, ci_upper = model.conf_int().loc[param_name]
                
#                 results.append({
#                     'Model': method.title(),
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
#     """Generate complete results section with all tables auto-updating from model_results."""
    
#     # Generate all tables
#     table1_md = df_to_markdown(create_selection_summary_table(model_results, response_variables))
#     table2_md = df_to_markdown(create_variable_selection_table(model_results, 'Surgical Length of Stay'))
#     table3_md = df_to_markdown(extract_treatment_effects_by_outcome(model_results, 'Surgical Length of Stay'))
#     table4_md = df_to_markdown(extract_covariate_effects(model_results, 'Surgical Length of Stay', 'univariate'))
#     table5_md = df_to_markdown(extract_treatment_effects_by_outcome(model_results, 'Postop Sepsis Occurrence'))
#     table6_md = df_to_markdown(extract_treatment_effects_by_outcome(model_results, 'Readmission likely related to Primary Procedure'))
    
#     results_markdown = f"""
# ## Results

# ### Overview of Variable Selection and Modeling

# We fit regression models for four postoperative outcomes using three variable selection approaches: univariate testing (p < 0.1), elastic net regularization (L1 ratio = 0.5, 5-fold cross-validation), and consensus selection (variables selected by both methods). The treatment variable (incisional negative pressure wound therapy, INPWT) was included in all models regardless of selection results. Due to small event counts relative to the number of selected predictors, several models could not be fit while maintaining the guideline of 10 events per variable.

# **Table 1. Variable Selection Summary by Outcome and Method**

# {table1_md}

# ### Surgical Length of Stay

# All three variable selection approaches yielded models that converged successfully. The univariate method selected 5 covariates (hospital discharge destination, diabetes status, hypertension, primary payor, and age). Elastic net selected 7 covariates, including 3 that overlapped with univariate selection. The consensus model included only the 3 variables selected by both methods.

# **Table 2. Variables Selected for Surgical Length of Stay Models**

# {table2_md}

# The treatment effect remained consistent across all three models, showing a modest reduction in length of stay that approached statistical significance in the elastic net and consensus models (p = 0.080 and p = 0.083, respectively).

# **Table 3. Treatment Effect on Surgical Length of Stay Across Models**

# {table3_md}

# INPWT was associated with a 9–13% reduction in expected length of stay across models (IRR 0.87–0.91), though this did not reach conventional statistical significance in the univariate model (p = 0.216). The elastic net model, which had the smallest sample size due to missing data in laboratory covariates, showed the largest effect estimate (IRR 0.87, 95% CI 0.74–1.02, p = 0.080).

# Other significant predictors of length of stay included discharge destination and diabetes status. Patients discharged to skilled care facilities or other non-home destinations had 35–64% longer stays (IRR 1.35–1.64, p < 0.01 across models). Patients without diabetes or with non-insulin-dependent diabetes had approximately 28% shorter stays compared to those with insulin-dependent diabetes (IRR 0.72, 95% CI 0.58–0.89, p < 0.004) in the univariate model. In the elastic net model, higher serum creatinine was associated with longer stays (IRR 1.35 per unit increase, 95% CI 1.11–1.63, p = 0.002), and White race was associated with shorter stays compared to other racial/ethnic groups (IRR 0.78, 95% CI 0.64–0.96, p = 0.017).

# **Table 4. Key Covariate Effects on Surgical Length of Stay (Univariate Model)**

# {table4_md}

# ### Postoperative Sepsis

# Neither univariate testing nor elastic net regularization identified any covariates associated with postoperative sepsis at the specified thresholds. All three models therefore included only the treatment variable and were identical in specification and results.

# **Table 5. Treatment Effect on Postoperative Sepsis (All Models)**

# {table5_md}

# INPWT demonstrated a significant protective association with postoperative sepsis, with treated patients showing 64% lower odds of developing sepsis (OR 0.36, 95% CI 0.16–0.85, p = 0.019). This effect remained robust across all modeling approaches. The absence of selected covariates suggests that, within this dataset, the treatment effect on sepsis was not confounded by measured patient or procedural characteristics.

# ### Postoperative Surgical Site Infection

# The small number of SSI events (n = 26) precluded fitting multivariable models while maintaining adequate events per variable. Univariate selection identified 2 covariates (presence of organ/space SSI at time of surgery, CPT procedure code), but including these with the treatment variable would have resulted in fewer than 10 events per predictor. Elastic net selected 13 covariates, far exceeding sample size limitations. No models were fit for this outcome due to insufficient statistical power.

# ### Readmission Related to Primary Procedure

# With only 23 readmission events, univariate selection of 7 covariates (organ/space SSI at time of surgery, BUN, tobacco use, serum creatinine, functional health status, PTT, and hematocrit) exceeded the events-per-variable guideline. Elastic net selected no covariates, allowing a treatment-only model to be fit.

# **Table 6. Treatment Effect on Readmission (Treatment-Only Model)**

# {table6_md}

# No evidence of a treatment effect on readmission was observed (OR 0.98, 95% CI 0.40–2.41, p = 0.959). The pseudo R-squared was near zero (0.00002), indicating the treatment explained essentially none of the variation in readmission risk. This null finding should be interpreted cautiously given the small number of events and lack of covariate adjustment.

# ### Summary

# INPWT showed a consistent protective association with postoperative sepsis across all modeling approaches, with a robust 64% reduction in odds (OR 0.36, p = 0.019). This effect was not confounded by measured patient characteristics. For surgical length of stay, INPWT demonstrated a modest benefit (9–13% reduction) that approached statistical significance in the elastic net and consensus models (p = 0.080 and p = 0.083, respectively), with point estimates consistently below 1.0 across all three approaches. The similarity of treatment effects across different covariate adjustment strategies suggests these findings are not sensitive to variable selection methods.

# Discharge disposition and diabetes status were stronger determinants of length of stay than the intervention itself, with discharge to non-home settings associated with 35–64% longer stays and absence of insulin-dependent diabetes associated with 28% shorter stays. Laboratory markers, particularly serum creatinine, also predicted prolonged hospitalization in the elastic net model.

# Insufficient events precluded definitive analysis of SSI (26 events) and readmission (23 events) outcomes. The sample size limitations highlight the challenge of multivariable modeling with rare outcomes in surgical cohorts and suggest these endpoints may require larger samples or alternative analytical approaches for adequate power.
# """
    
#     return results_markdown


# # # USAGE
# # # Generate and print
# # # display(generate_results_section(model_results, response_variables))
# # results_markdown = generate_results_section(model_results, response_variables)
# # print(results_markdown)  # Copy this output and paste into a markdown cell


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
                
            if 'logistic_regression' in model_data:
                model = model_data['logistic_regression'].model
                n_obs = int(model.nobs)
                
                # Count events for binary outcomes
                y = model_data['logistic_regression'].y
                n_events = int(y.sum())
                
                # Find INPWT coefficient
                inpwt_params = [p for p in model.params.index if 'INPWT' in p]
                if inpwt_params:
                    param_name = inpwt_params[0]
                    coef = model.params[param_name]
                    pval = model.pvalues[param_name]
                    ci_lower, ci_upper = model.conf_int().loc[param_name]
                    
                    results.append({
                        'Outcome': outcome,
                        'Method': method.replace('_', ' ').title(),
                        'n': n_obs,
                        'Events': n_events,
                        'Coefficient': round(coef, 3),
                        'OR': round(np.exp(coef), 2),
                        '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                        'p-value': f"{pval:.3f}"
                    })
                    
            elif 'negative_binomial' in model_data:
                model = model_data['negative_binomial'].model
                n_obs = int(model.nobs)
                
                # Find INPWT coefficient
                inpwt_params = [p for p in model.params.index if 'INPWT' in p]
                if inpwt_params:
                    param_name = inpwt_params[0]
                    coef = model.params[param_name]
                    pval = model.pvalues[param_name]
                    ci_lower, ci_upper = model.conf_int().loc[param_name]
                    
                    results.append({
                        'Outcome': outcome,
                        'Method': method.replace('_', ' ').title(),
                        'n': n_obs,
                        'Coefficient': round(coef, 3),
                        'IRR': round(np.exp(coef), 2),
                        '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                        'p-value': f"{pval:.3f}"
                    })
    
    return pd.DataFrame(results)


def create_selection_summary_table(model_results, response_variables):
    """Table 1: Variable Selection Summary by Outcome and Method"""
    results = []
    
    for outcome in response_variables:
        if outcome not in model_results:
            continue
            
        for method in ['univariate', 'elastic_net', 'consensus']:
            if method in model_results[outcome] and model_results[outcome][method] is not None:
                model_data = model_results[outcome][method]
                # Subtract 1 for treatment variable to get covariate count
                n_covariates = len(model_data['selected_variables']) - 1
                
                results.append({
                    'Outcome': outcome,
                    'Method': method.replace('_', ' ').title(),
                    'Covariates Selected': n_covariates,
                    'Model Converged': '✓',
                    'Notes': '—'
                })
            else:
                # Model was attempted but failed to converge
                results.append({
                    'Outcome': outcome,
                    'Method': method.replace('_', ' ').title(),
                    'Covariates Selected': '—',
                    'Model Converged': '✗',
                    'Notes': 'Failed to converge'
                })
    
    return pd.DataFrame(results)


def create_variable_selection_table(model_results, outcome):
    """Table showing which variables were selected by each method for a specific outcome."""
    if outcome not in model_results:
        return pd.DataFrame()
    
    all_vars = set()
    method_vars = {}
    
    for method, model_data in model_results[outcome].items():
        if model_data is not None:
            vars_selected = model_data['selected_variables']
            method_vars[method] = set(vars_selected)
            all_vars.update(vars_selected)
    
    if len(all_vars) == 0:
        return pd.DataFrame()
    
    # Sort: INPWT first, then alphabetical
    sorted_vars = sorted(all_vars, key=lambda x: (x != 'INPWT', x))
    
    results = []
    for var in sorted_vars:
        row = {'Variable': var}
        for method in ['univariate', 'elastic_net', 'consensus']:
            if method in method_vars:
                row[method.replace('_', ' ').title()] = '✓' if var in method_vars[method] else ''
            else:
                row[method.replace('_', ' ').title()] = '—'
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
            
        if 'logistic_regression' in model_data:
            model = model_data['logistic_regression'].model
            n_obs = int(model.nobs)
            y = model_data['logistic_regression'].y
            n_events = int(y.sum())
            
            inpwt_params = [p for p in model.params.index if 'INPWT' in p]
            if inpwt_params:
                param_name = inpwt_params[0]
                coef = model.params[param_name]
                pval = model.pvalues[param_name]
                ci_lower, ci_upper = model.conf_int().loc[param_name]
                
                results.append({
                    'Model': method.replace('_', ' ').title(),
                    'n': n_obs,
                    'Events': n_events,
                    'Coefficient': round(coef, 3),
                    'OR': round(np.exp(coef), 2),
                    '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                    'p-value': f"{pval:.3f}"
                })
                
        elif 'negative_binomial' in model_data:
            model = model_data['negative_binomial'].model
            n_obs = int(model.nobs)
            
            inpwt_params = [p for p in model.params.index if 'INPWT' in p]
            if inpwt_params:
                param_name = inpwt_params[0]
                coef = model.params[param_name]
                pval = model.pvalues[param_name]
                ci_lower, ci_upper = model.conf_int().loc[param_name]
                
                results.append({
                    'Model': method.replace('_', ' ').title(),
                    'n': n_obs,
                    'Coefficient': round(coef, 3),
                    'IRR': round(np.exp(coef), 2),
                    '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
                    'p-value': f"{pval:.3f}"
                })
    
    return pd.DataFrame(results)


def extract_covariate_effects(model_results, outcome, method='univariate', top_n=None):
    """Extract covariate effects from a specific model."""
    if outcome not in model_results or method not in model_results[outcome]:
        return pd.DataFrame()
    
    model_data = model_results[outcome][method]
    if model_data is None:
        return pd.DataFrame()
    
    if 'logistic_regression' in model_data:
        model = model_data['logistic_regression'].model
        effect_type = 'OR'
    elif 'negative_binomial' in model_data:
        model = model_data['negative_binomial'].model
        effect_type = 'IRR'
    else:
        return pd.DataFrame()
    
    results = []
    for param_name in model.params.index:
        if param_name == 'const' or param_name == 'alpha' or 'INPWT' in param_name:
            continue
            
        coef = model.params[param_name]
        pval = model.pvalues[param_name]
        ci_lower, ci_upper = model.conf_int().loc[param_name]
        
        results.append({
            'Predictor': param_name,
            'Coefficient': round(coef, 3),
            effect_type: round(np.exp(coef), 2),
            '95% CI': f"{np.exp(ci_lower):.2f} – {np.exp(ci_upper):.2f}",
            'p-value': f"{pval:.3f}"
        })
    
    df = pd.DataFrame(results)
    
    # Sort by p-value and optionally take top N
    if len(df) > 0:
        df['p_numeric'] = df['p-value'].astype(float)
        df = df.sort_values('p_numeric')
        df = df.drop('p_numeric', axis=1)
        
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
    
    # Check which outcomes have converged models to report on
    outcomes_to_report = {}
    for outcome in response_variables:
        if outcome in model_results:
            converged_methods = [m for m, d in model_results[outcome].items() if d is not None]
            if converged_methods:
                outcomes_to_report[outcome] = converged_methods
    
    results_markdown = f"""
## Results

### Overview of Variable Selection and Modeling

We employed a systematic two-stage approach to variable selection and model fitting. First, we applied two independent variable selection methods—univariate testing (p < 0.1) and elastic net regularization (L1 ratio = 0.5, 5-fold cross-validation)—to identify candidate predictors for each outcome. Second, we determined the maximum number of covariates that could be reliably estimated given our sample size, using established guidelines: 10 events per variable for binary outcomes (Peduzzi et al., 1996) and 15 observations per variable for count outcomes, the latter being more conservative to account for increased variance in count data.

When either selection method identified more variables than the sample size threshold permitted, we retained only the top N variables ranked by statistical strength (smallest p-values for univariate, largest absolute coefficients for elastic net). The treatment variable (incisional negative pressure wound therapy, INPWT) was included in all models regardless of selection and counted toward the variable limit.

We attempted to fit three models per outcome: (1) univariate selection, (2) elastic net selection, and (3) consensus (variables selected by both methods). For outcomes where models failed to converge despite appropriate sample size considerations, we concluded the data were insufficient to support that covariate structure and report this limitation explicitly.

**Table 1. Variable Selection and Model Convergence by Outcome and Method**

{table1_md}

"""

    # Add outcome-specific sections
    
    # ========================================================================
    # SURGICAL LENGTH OF STAY
    # ========================================================================
    if 'Surgical Length of Stay' in outcomes_to_report:
        var_table = create_variable_selection_table(model_results, 'Surgical Length of Stay')
        var_table_md = df_to_markdown(var_table)
        
        treatment_table = extract_treatment_effects_by_outcome(model_results, 'Surgical Length of Stay')
        treatment_table_md = df_to_markdown(treatment_table)
        
        covariate_table = extract_covariate_effects(model_results, 'Surgical Length of Stay', 'univariate', top_n=10)
        covariate_table_md = df_to_markdown(covariate_table)
        
        results_markdown += f"""
### Surgical Length of Stay

The univariate method (p < 0.1) selected 6 covariates, all within the sample size limit of 9 covariates (152 observations / 15 = 10 total variables, minus 1 for treatment). Elastic net selected 13 covariates; we retained the top 9 by coefficient magnitude. However, 6 of these 9 variables exhibited perfect separation (certain predictor levels had 100% of observations at specific outcome values), and the model failed to converge. The consensus approach was not applicable as there was no overlap between univariate and elastic net selections after EPV limiting. Thus, only the univariate model converged successfully.

**Variables in Univariate Model**

{var_table_md}

Two variables exhibited perfect separation (Hospital Discharge Destination: Acute Care Hospital; Duration of Surgical Procedure: 29 minutes), yet the univariate model converged successfully. This suggests these patterns, while extreme, did not prevent parameter estimation in this specific covariate configuration.

**Treatment Effect on Surgical Length of Stay**

{treatment_table_md}

INPWT was associated with a 9% reduction in expected length of stay (IRR 0.91, 95% CI 0.78–1.05), though this did not reach statistical significance (p = 0.190).

**Key Covariate Effects (Univariate Model)**

{covariate_table_md}

Discharge destination was the strongest predictor of length of stay. Patients discharged to other facilities (not home or skilled care) had 42% longer stays (IRR 1.35, 95% CI 1.14–1.75, p = 0.001), and those discharged to skilled care facilities had 33% longer stays (IRR 1.33, 95% CI 1.06–1.67, p = 0.015), compared to discharge home. 

Diabetes status also significantly predicted length of stay. Patients without diabetes had 27% shorter stays compared to insulin-dependent diabetics (IRR 0.73, 95% CI 0.60–0.87, p < 0.001), and those with non-insulin-dependent diabetes had 27% shorter stays (IRR 0.73, 95% CI 0.58–0.90, p = 0.004).

"""
    
    # ========================================================================
    # POSTOPERATIVE SEPSIS
    # ========================================================================
    if 'Postop Sepsis Occurrence' in outcomes_to_report:
        treatment_table = extract_treatment_effects_by_outcome(model_results, 'Postop Sepsis Occurrence')
        treatment_table_md = df_to_markdown(treatment_table)
        
        results_markdown += f"""
### Postoperative Sepsis

With only 27 sepsis events, the events-per-variable guideline (10 events per variable) permitted a maximum of 2 total variables (treatment + 1 covariate). Neither univariate testing (p < 0.1) nor elastic net regularization identified any covariates. All three models (univariate, elastic net, and consensus) were therefore identical, consisting of the treatment variable only.

**Treatment Effect on Postoperative Sepsis (All Models)**

{treatment_table_md}

INPWT demonstrated a significant protective association with postoperative sepsis. Patients receiving INPWT had 64% lower odds of developing sepsis (OR 0.36, 95% CI 0.16–0.85, p = 0.019). This effect remained robust across all modeling approaches. The absence of selected covariates suggests that, within this dataset and given the statistical power constraints, the treatment effect on sepsis was not confounded by the measured patient or procedural characteristics that were available for analysis.

"""
    
    # ========================================================================
    # POSTOPERATIVE SSI
    # ========================================================================
    if 'Postop SSI Occurrence' in model_results:
        results_markdown += """
### Postoperative Surgical Site Infection

With only 26 SSI events, the events-per-variable guideline permitted a maximum of 2 total variables (treatment + 1 covariate). Both univariate testing and elastic net selection identified the same covariate: number of postoperative organ/space SSIs at time of surgery. However, this variable exhibited perfect separation—all patients with organ/space SSI at surgery also had subsequent SSI. This perfect collinearity prevented model convergence for both univariate and elastic net approaches.

The elastic net initially selected 14 variables but was limited to the top 1 by coefficient magnitude, which happened to be the same separated variable. No multivariable models could be fit for this outcome due to the separation issue combined with insufficient events for including additional predictors.

"""
    
    # ========================================================================
    # READMISSION
    # ========================================================================
    if 'Readmission likely related to Primary Procedure' in outcomes_to_report:
        treatment_table = extract_treatment_effects_by_outcome(model_results, 'Readmission likely related to Primary Procedure')
        treatment_table_md = df_to_markdown(treatment_table)
        
        results_markdown += f"""
### Readmission Related to Primary Procedure

With only 23 readmission events, the sample size permitted a maximum of 2 total variables (treatment + 1 covariate). Univariate testing identified 8 potentially associated covariates but was limited to the top 1 by p-value: number of postoperative organ/space SSIs. This variable exhibited perfect separation, causing the univariate model to fail to converge.

Elastic net selected no covariates, allowing a treatment-only model to be fit successfully.

**Treatment Effect on Readmission (Elastic Net Model)**

{treatment_table_md}

No evidence of a treatment effect on readmission was observed (OR 0.98, 95% CI 0.40–2.41, p = 0.959). The pseudo R-squared was near zero (0.00002), indicating the treatment explained essentially none of the variation in readmission risk. This null finding should be interpreted cautiously given the small number of events and lack of covariate adjustment.

"""
    
    results_markdown += """
### Summary

INPWT showed a significant protective association with postoperative sepsis, with a 64% reduction in odds (OR 0.36, 95% CI 0.16–0.85, p = 0.019). This effect was not confounded by measured patient characteristics that could be included given sample size constraints. For surgical length of stay, INPWT demonstrated a modest 9% reduction (IRR 0.91) that did not reach statistical significance (p = 0.190).

Discharge disposition and diabetes status were the strongest determinants of length of stay in the univariate model. Discharge to non-home settings was associated with 33–42% longer stays, and absence of insulin-dependent diabetes was associated with 27% shorter stays compared to insulin-dependent diabetes.

Sample size limitations precluded definitive analysis of SSI (26 events) and readmission (23 events) outcomes. The small event counts relative to the number of candidate predictors resulted in models that either could not be fit due to separation issues (SSI, readmission univariate) or had insufficient power for covariate adjustment (readmission elastic net, treatment-only model). The elastic net selection for surgical length of stay identified 13 variables, but including the top 9 by coefficient magnitude resulted in convergence failure due to separation in 6 variables, highlighting the challenge of multivariable modeling with moderate sample sizes and complex covariate patterns.

These findings underscore the importance of sample size considerations in surgical outcomes research. Future studies of these endpoints may require larger samples or alternative analytical approaches such as penalized regression methods specifically designed to handle separation, propensity score methods to reduce dimensionality, or Bayesian approaches that can incorporate prior information to stabilize estimates with sparse data.
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


# # pip install jupyter-book
# # jupyter-book create book
# # cp analysis.ipynb book/analysis.ipynb
# # cp -r statwise book/statwise

# # edit _toc.yml
# # format: jb-book
# # root: intro
# # chapters:
# #   - file: analysis
# #     title: INPWT Analysis

# # edit intro.md

# # cd book
# # jupyter-book clean . --all
# # jupyter-book build .
# # open _build/html/index.html

# rm -rf docs
# cp -r book/_build/html docs
# touch docs/.nojekyll

# git add docs
# git commit -m "Update docs with new build"
# git push