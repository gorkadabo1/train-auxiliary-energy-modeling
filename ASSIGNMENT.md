# Assignment: Modeling Energy Consumed by Auxiliary Equipment

## Objective

Modeling the energy consumed by auxiliary equipment as a function of the operating times of HVAC components.


## Background Information

### Context
- The consuming equipment connected to the AC branch of the auxiliary converter consists mainly of air production compressors and HVAC systems for both passenger compartments and driver cabins.
- One month of data aggregated by event type (interstation/stop/maintenance).

### Dataset Variables

| Variable | Description |
|----------|-------------|
| `date_time` | Date in format YYYY-MM-DD HH:MM:SS |
| `ut` | Train unit identifier |
| `eaux_train_ac` | Energy consumed by AC branch [kWh] |
| `event` | Event type (interstation/stop/maintenance) |
| `event_duration` | Event duration [s] |
| `suma_ton_compresores` | Sum of ON-time for train compressors [s] |
| `suma_hvac_ton_compresores` | Sum of ON-time for saloon HVAC compressors [s] |
| `suma_hvac_ton_heater` | Sum of ON-time for saloon HVAC heaters [s] |
| `suma_ton_comp_cab` | Sum of ON-time for cabin HVAC compressors [s] |
| `suma_ton_heater_cab` | Sum of ON-time for cabin HVAC heaters [s] |

### Mathematical Model

```
y ~ β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

Where:
- **y**: Dependent variable (AC auxiliary energy per event) [kWh]
- **x₁..ₙ**: Independent/explanatory variables (operating times of auxiliary components) [s]
- **β₁..ₙ**: Regression coefficients indicating the weight of each independent variable (average power estimated by the model)
- **β₀**: Regression coefficient introducing a bias to the system (common magnitude for all subjects)

---

## Tasks

### 1) Load the data "data.RData"

### 2) Filter the data
- Remove events longer than 600 seconds
- Remove maintenance events ("maintenance")

### 3) Consumption comparison
- Is there a significant difference in consumption (`eaux_train_ac`) between "interstation" and "stop" event types? Justify your answer.
- Is there a significant difference in power (kW) between "interstation" and "stop" event types? Justify your answer.

### 4) Generate a linear regression model for consumption (eaux_train_ac) using the first fortnight of data 
Model the auxiliary consumption with the filtered data (from step 3).

**NOTE:** Do not include the effect of time or train unit (ut) in the model.

### 5) Review of the generated regression model 
- Is the generated model explanatory? Justify your answer.
- Does it make sense to work with a reduced model or stratified models? Justify your answer.
- List the variables that are statistically significant.
- Explain the meaning of each regression coefficient.
- Estimate the consumption in kW for each regression coefficient.
- Plot the distribution of residuals. Are there any anomalies? If so, can they be explained? Justify your answer.

### 6) Validate the regression model generated in step 4) with the second fortnight of September data 
- Use RMSE as the evaluation metric.
- Does the regression model generalize well?
- Is the error obtained in training and validation similar?

### 7) Generate a more complex ML predictive model with the training subset and validate it on the validation subset defined previously 
- Compare the new RMSE metrics in training and validation of the new ML model with the linear regression model from step 4).
- Can the predictive capacity for consumption be improved compared to the linear model from step 4)?
- What happens with variable interpretability in the new model?
- Select the final model to deploy. Justify your answer.
