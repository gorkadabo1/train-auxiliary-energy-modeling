# ==============================================================================
# Train Auxiliary Energy Consumption Modeling
# Author: Gorka Dabó
# ==============================================================================

# 0) Libraries ####
library(ggplot2)
library(dplyr)
library(effsize)
library(lubridate)
library(randomForest)
library(ROCR)

# 1) Data Loading ####
load("data/data.RData")

# Quick data checks
str(df)
dim(df)
anyNA(df)

# 2) Data Filtering ####
# Remove maintenance events and events longer than 600 seconds
df_clean <- df %>%
  filter(event != "MAINTENANCE",
         event_duration <= 600)

# Calculate power in kW (energy/time conversion)
df_clean <- df_clean %>%
  mutate(power_kW = eaux_train_ac / (event_duration / 3600))


# 3) Consumption Comparison: interstation vs stop ####

## a) Energy consumption (eaux_train_ac) comparison #####
df_inter <- df_clean %>% 
  filter(event == "interstation")

df_stop <- df_clean %>% 
  filter(event == "stop")

# Shapiro-Wilk normality test
# Note: shapiro.test accepts sample sizes between 3 and 5000
# For groups with >5000 observations, we take a random sample
set.seed(123)
sh1 <- shapiro.test(sample(df_inter$eaux_train_ac, 5000))
sh2 <- shapiro.test(sample(df_stop$eaux_train_ac, 5000))
sh1; sh2

# Observation: p-values are practically 0.
# With large n, the Shapiro test is very sensitive and may yield p < 0.05 
# even for slight deviations. However, in this dataset, the obtained W values 
# (~0.8, ~0.4) are very low, indicating that the data do not fit a normal 
# distribution well. This result is not attributable solely to sample size.

# Visualize distributions
df_plot <- df_clean %>%
  filter(event %in% c("interstation", "stop")) %>%
  select(event, eaux_train_ac) %>%
  mutate(event = factor(event, levels = c("interstation", "stop"))) %>%
  tidyr::drop_na()

ggplot(df_plot, aes(x = event, y = eaux_train_ac)) +
  geom_boxplot(outlier.alpha = 0.35) +
  labs(
    title = "Boxplots of eaux_train_ac by Event Type",
    x = "Event",
    y = "eaux_train_ac"
  ) +
  theme_minimal()

# Histogram for "interstation"
ggplot(df_plot %>% filter(event == "interstation"),
       aes(x = eaux_train_ac)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 50) +
  labs(
    title = "Histogram of eaux_train_ac - Event: interstation",
    x = "eaux_train_ac",
    y = "Frequency"
  ) +
  theme_minimal()

# Histogram for "stop"
ggplot(df_plot %>% filter(event == "stop"),
       aes(x = eaux_train_ac)) +
  geom_histogram(fill = "salmon", color = "black", bins = 50) +
  labs(
    title = "Histogram of eaux_train_ac - Event: stop",
    x = "eaux_train_ac",
    y = "Frequency"
  ) +
  theme_minimal()

# Both boxplots show long upper tails, and both histograms display skewed,
# non-normal distributions. We conclude that consumption distributions 
# between events do not follow a normal distribution.

# Since data are not normal, we use Wilcoxon test to compare groups
wilcox_res <- wilcox.test(df_inter$eaux_train_ac, df_stop$eaux_train_ac, exact = FALSE)
wilcox_res

# Result shows p-value ≈ 0, indicating statistically significant difference.

# However, with very large n, the test is sensitive and may detect minimal 
# differences as significant. We calculate effect size to assess practical importance.
cd <- effsize::cliff.delta(df_inter$eaux_train_ac, df_stop$eaux_train_ac)
cd

# Result (δ ≈ 0.82, "large" effect) indicates the difference is not only
# statistically significant but also very large in practical magnitude:
# interstation values are mostly much higher than stop values.
# CONCLUSION: There IS a significant difference in energy consumption between
# interstation and stop event types.

# ROC curve to quantify discriminatory capacity
df_roc <- df_clean %>%
  filter(event %in% c("interstation", "stop")) %>%
  transmute(
    eaux_train_ac,
    y = ifelse(event == "interstation", 1, 0)
  )

roc_df <- function(scores, labels) {
  pred <- ROCR::prediction(scores, labels)
  perf <- ROCR::performance(pred, measure = "tpr", x.measure = "fpr")
  auc  <- ROCR::performance(pred, "auc")@y.values[[1]]
  
  fpr  <- perf@x.values[[1]]
  tpr  <- perf@y.values[[1]]
  
  data.frame(fpr = fpr, tpr = tpr, auc = auc)
}

# ROC with eaux_train_ac
roc_eaux <- roc_df(df_roc$eaux_train_ac, df_roc$y)

# Plot ROC curve
ggplot(roc_eaux, aes(x = fpr, y = tpr)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "ROC: interstation vs stop (score = eaux_train_ac)",
    subtitle = paste0("AUC = ", round(unique(roc_eaux$auc), 3)),
    x = "FPR (1 - Specificity)",
    y = "TPR (Sensitivity)"
  ) +
  theme_minimal()

# AUC ≈ 0.91 shows excellent discriminatory capacity of eaux_train_ac
# to differentiate between interstation and stop events.


## b) Power (kW) comparison: interstation vs stop #####
# Shapiro-Wilk test for power
set.seed(123)
sh1 <- shapiro.test(sample(df_inter$power_kW, 5000))
sh2 <- shapiro.test(sample(df_stop$power_kW, 5000))
sh1; sh2

# p-values are practically 0, but W values are close to 0.95, indicating
# the data fit a normal distribution reasonably well despite low p-values.

# Visualize power distributions
df_plot <- df_clean %>%
  filter(event %in% c("interstation", "stop")) %>%
  select(event, power_kW) %>%
  mutate(event = factor(event, levels = c("interstation", "stop"))) %>%
  tidyr::drop_na()

ggplot(df_plot, aes(x = event, y = power_kW)) +
  geom_boxplot(outlier.alpha = 0.35) +
  labs(
    title = "Boxplots of power_kW by Event Type",
    x = "Event",
    y = "power_kW"
  ) +
  theme_minimal()

# Distributions appear visually symmetric and similar.
# Given the Central Limit Theorem and large n, we apply t-test.
t_res <- t.test(df_inter$power_kW, df_stop$power_kW, var.equal = TRUE)
t_res

# Result shows p-value ≈ 0, statistically significant.

# Calculate effect size
cd <- effsize::cliff.delta(df_inter$power_kW, df_stop$power_kW)
cd

# Result: Cliff's delta ≈ 0.08 indicates a practically null effect.
# Although statistically significant, the practical magnitude of the
# power difference between events is minimal.
# CONCLUSION: There is NO significant practical difference in power (kW)
# between interstation and stop event types.

# Confirm with ROC curve
df_roc <- df_clean %>%
  filter(event %in% c("interstation", "stop")) %>%
  transmute(power_kW, y = ifelse(event == "interstation", 1, 0))

roc_power <- roc_df(df_roc$power_kW, df_roc$y)

ggplot(roc_power, aes(x = fpr, y = tpr)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "ROC: interstation vs stop (score = power_kW)",
    subtitle = paste0("AUC = ", round(unique(roc_power$auc), 3)),
    x = "FPR (1 - Specificity)",
    y = "TPR (Sensitivity)"
  ) +
  theme_minimal()

# AUC ≈ 0.54 indicates power_kW barely discriminates between events,
# reinforcing that power differences are minimal and without predictive value.


# 4) Linear Regression Model ####

# Filter first fortnight (Sept 1-15, 2019)
df_quincena1 <- df_clean %>%
  filter(date_time >= ymd("2019-09-01"),
         date_time < ymd("2019-09-16"))

# Create linear model
# Dependent variable: eaux_train_ac
# Explanatory variables: event type, duration, HVAC and compressor operating times
modelo_lin <- lm(
  eaux_train_ac ~ suma_ton_compresores +
    suma_hvac_ton_compresores +
    suma_hvac_ton_heater +
    suma_ton_comp_cab +
    suma_ton_heater_cab +
    event_duration +
    as.factor(event),
  data = df_quincena1
)

# Model summary
summary(modelo_lin)


# 5) Model Review ####

## a) Is the model explanatory? ####
# The linear regression model achieves excellent fit:
# - R² ≈ 0.993: explains ~99% of energy consumption variability
# - All coefficients highly significant (p < 2e-16)
# - Including event_duration notably improves explanatory capacity
# - The event factor is also significant (consumption differs between event types)
# - Residuals are small and centered around zero without clear patterns
# CONCLUSION: YES, the model is highly explanatory.

## b) Reduced model or stratified models? ####
# In the global model summary:
# - All variables are highly significant (p < 2e-16)
# - event_duration provides substantial explanatory power
# - Operating times remain relevant
# Since all contribute useful information, no variable elimination is warranted.

# Check multicollinearity via correlation matrix
preds <- c("suma_ton_compresores","suma_hvac_ton_compresores",
           "suma_hvac_ton_heater","suma_ton_comp_cab",
           "suma_ton_heater_cab","event_duration")

round(cor(df_quincena1[, preds], use = "pairwise.complete.obs"), 3)

# Highest correlation with event_duration (~0.72 with suma_ton_compresores)
# Other correlations are low/moderate (≤ ~0.41)
# No severe multicollinearity detected.
# CONCLUSION: No reduced model needed; all variables contribute without 
# concerning collinearity.

# Test stratified models by event type
df_inter_event <- df_quincena1 %>% filter(event == "interstation")
df_stop_event  <- df_quincena1 %>% filter(event == "stop")

modelo_inter <- lm(
  eaux_train_ac ~ suma_ton_compresores +
    suma_hvac_ton_compresores +
    suma_hvac_ton_heater +
    suma_ton_comp_cab +
    suma_ton_heater_cab + 
    event_duration,
  data = df_inter_event
)

modelo_stop <- lm(
  eaux_train_ac ~ suma_ton_compresores +
    suma_hvac_ton_compresores +
    suma_hvac_ton_heater +
    suma_ton_comp_cab +
    suma_ton_heater_cab +
    event_duration,
  data = df_stop_event
)

summary(modelo_inter)
summary(modelo_stop)

# Compare R² and RSE
r2_global  <- summary(modelo_lin)$r.squared
rse_global <- summary(modelo_lin)$sigma

cat("GLOBAL Model  -> R² =", round(r2_global, 4), " | RSE =", round(rse_global, 4), "\n")
cat("interstation  -> R² =", round(summary(modelo_inter)$r.squared, 4),
    " | RSE =", round(summary(modelo_inter)$sigma, 4), "\n")
cat("stop          -> R² =", round(summary(modelo_stop)$r.squared, 4),
    " | RSE =", round(summary(modelo_stop)$sigma, 4), "\n")

# The global model already captures level differences between events.
# Its R² (~0.9927) is as high as the best stratum, and RSE (~0.062)
# falls between stop (very low, 0.0315) and interstation (higher, 0.0804).
# Separating by strata provides no clear global improvement and complicates 
# maintenance with two models.
# CONCLUSION: No need for reduced or stratified models.

## c) Statistically significant variables ####
# In the global model, all variables obtain p < 2e-16:
#   - (Intercept)              
#   - suma_ton_compresores     
#   - suma_hvac_ton_compresores 
#   - suma_hvac_ton_heater     
#   - suma_ton_comp_cab        
#   - suma_ton_heater_cab      
#   - event_duration            
#   - as.factor(event)stop

## d) Meaning of each regression coefficient ####
# Each coefficient represents the average consumption increase (in kWh)
# associated with a one-unit increase in the corresponding variable,
# holding other variables constant.
#
# - (Intercept): Average base consumption when equipment is off and event is "interstation"
# - suma_ton_compresores: Average consumption increase per additional second of main compressor operation
# - suma_hvac_ton_compresores: Consumption increase from HVAC compressor operation
# - suma_hvac_ton_heater: Average consumption increase per second of HVAC heater operation
# - suma_ton_comp_cab: Consumption increase per additional second of cabin compressor operation
# - suma_ton_heater_cab: Average consumption increase per second of cabin heater operation
# - event_duration: Effect of total event time on consumption (longer duration = higher total consumption)
# - as.factor(event)stop: Average consumption change when switching from "interstation" to "stop" event

## e) Estimate consumption in kW for each coefficient ####
# Model coefficients are in kWh/s, so we convert to kW by multiplying by 3600
coefs <- coef(modelo_lin)
coefs_kW <- coefs * 3600
round(coefs_kW, 2)

# Results in kW:
# (Intercept)               → -55.73 kW  → Negative base consumption; adjustment to center regression
# suma_ton_compresores      → 12.67 kW   → Average power from main compressors
# suma_hvac_ton_compresores → 7.82 kW    → Average power from HVAC compressors
# suma_hvac_ton_heater      → 9.84 kW    → Average power from HVAC heaters
# suma_ton_comp_cab         → 2.39 kW    → Average power from cabin compressors
# suma_ton_heater_cab       → 1.52 kW    → Average power from cabin heaters
# event_duration            → 12.30 kW   → Equivalent average power from event duration effect
# as.factor(event)stop      → 17.26 kW   → Average power increase for "stop" events vs "interstation"


## f) Residual Analysis ####

# Extract residuals and fitted values
residuos <- resid(modelo_lin)
ajustados <- fitted(modelo_lin)

# QQ-plot of residuals
qqnorm(residuos, main = "QQ-Plot of Residuals")
qqline(residuos, col = "red", lwd = 2)
# Central points follow the line well, but tails deviate slightly,
# indicating slightly heavier tails than a normal distribution.

# Observed vs Predicted values
df_pred <- data.frame(
  observado = df_quincena1$eaux_train_ac,
  predicho  = ajustados
)

ggplot(df_pred, aes(x = predicho, y = observado)) +
  geom_point(alpha = 0.3, color = "#2c7fb8") +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1) +
  labs(
    title = "Observed vs Predicted Values (Linear Model)",
    subtitle = "Red line represents perfect fit (y = ŷ)",
    x = "Predicted (ŷ)",
    y = "Observed (y)"
  ) +
  theme_minimal()
# Nearly perfect alignment confirms the excellent model fit (R² ≈ 0.993)

# Residuals vs train unit
ggplot(df_quincena1, aes(x = as.factor(ut), y = residuos)) +
  geom_boxplot(outlier.alpha = 0.3, fill = "skyblue") +
  labs(
    title = "Residual Distribution by Train Unit (ut)",
    x = "Train Unit",
    y = "Residual"
  ) +
  theme_minimal()
# Most trains have residuals centered at 0, but train 29 shows 
# consistently positive residuals (~3), indicating different behavior 
# or sensor deviation.

# Residuals vs time (chronological order)
ggplot(df_quincena1, aes(x = date_time, y = residuos)) +
  geom_point(alpha = 0.3, color = "#2c7fb8") +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  labs(
    title = "Residuals Over Time",
    x = "Date and Time",
    y = "Residual"
  ) +
  theme_minimal()
# Residuals remain centered at 0 throughout the period without clear patterns.
# Time does not appear to influence residuals.

# ANOVA on train unit
anova_ut <- aov(residuos ~ as.factor(ut), data = df_quincena1)
summary(anova_ut)
# Result: p < 2e-16 → significant differences → train unit influences residuals.

# Extended models to confirm which variable explains the anomaly
modelo_ut <- update(modelo_lin, . ~ . + as.factor(ut))
modelo_time <- update(modelo_lin, . ~ . + as.numeric(date_time))

# Compare adjusted R²
summary(modelo_lin)$adj.r.squared   # 0.9927
summary(modelo_ut)$adj.r.squared    # 0.9952 → ut explains the anomaly
summary(modelo_time)$adj.r.squared  # No change → time doesn't affect error

# RESIDUAL SUMMARY:
# - Residuals are highly concentrated around 0 with symmetric shape
# - Majority of errors are very small, with few extreme values
# - No systematic patterns indicating model misspecification
# - The model meets homoscedasticity and zero-mean assumptions reasonably well
# - Train unit 29 shows anomalous behavior (could be sensor issue)


# 6) Validation with Second Fortnight (RMSE) ####
# Filter validation window: Sept 16-30, 2019
df_quincena2 <- df_clean %>%
  filter(date_time >= lubridate::ymd("2019-09-16"),
         date_time <  lubridate::ymd("2019-10-01"))

# Predictions using the model trained on first fortnight
df_quincena2$pred <- predict(modelo_lin, newdata = df_quincena2)

# Global RMSE
rmse_val <- sqrt(mean((df_quincena2$eaux_train_ac - df_quincena2$pred)^2))
rmse_val

# RMSE by event type
rmse_por_evento <- df_quincena2 %>%
  mutate(se = (eaux_train_ac - pred)^2) %>%
  group_by(event) %>%
  summarise(RMSE = sqrt(mean(se)), .groups = "drop")
rmse_por_evento

# Predicted vs Observed plot
ggplot(df_quincena2, aes(x = pred, y = eaux_train_ac)) +
  geom_point(alpha = 0.25) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Validation (2nd Fortnight): Predicted vs Observed",
       x = "Predicted (kWh)", y = "Observed (kWh)") +
  theme_minimal()

# RESULTS:
# - Validation RMSE (≈ 0.059 kWh) is very low and similar to training residual error (≈ 0.062 kWh)
# - This demonstrates the model generalizes very well with no overfitting
# - RMSE by event type: interstation = 0.0748, stop = 0.0351
# - Higher error for interstation is logical (more variable consumption during movement)
# - The Predicted vs Observed plot shows near-perfect alignment with diagonal


# 7) ML Predictive Model ####

## a) Compare metrics ####
preds <- c("suma_ton_compresores","suma_hvac_ton_compresores",
           "suma_hvac_ton_heater","suma_ton_comp_cab",
           "suma_ton_heater_cab","event_duration","event")

df_quincena1 <- df_quincena1 %>% 
  mutate(event = as.factor(event))
df_quincena2 <- df_quincena2 %>% 
  mutate(event = as.factor(event))

train_df <- df_quincena1 %>%
  select(eaux_train_ac, all_of(preds))

valid_df <- df_quincena2 %>%
  select(eaux_train_ac, all_of(preds))

# Baseline: Linear model RMSE
rmse_lin_train <- sqrt(mean(resid(modelo_lin)^2))
pred_lin_valid <- predict(modelo_lin, newdata = valid_df)
rmse_lin_valid <- sqrt(mean((valid_df$eaux_train_ac - pred_lin_valid)^2))

cat("Baseline Linear Regression - RMSE train:", round(rmse_lin_train, 4),
    " | RMSE valid:", round(rmse_lin_valid, 4), "\n")

# Random Forest model
rf_model <- randomForest(
  eaux_train_ac ~ .,
  data      = train_df,
  ntree     = 200,
  mtry      = 3,
  nodesize  = 50,
  maxnodes  = 40,
  importance = TRUE,
  keep.inbag = TRUE
)

# Predictions and RMSE
pred_rf_train <- predict(rf_model, newdata = train_df[, preds])
pred_rf_valid <- predict(rf_model, newdata = valid_df[, preds])

rmse_rf_train <- sqrt(mean((train_df$eaux_train_ac - pred_rf_train)^2))
rmse_rf_valid <- sqrt(mean((valid_df$eaux_train_ac - pred_rf_valid)^2))

cat("Random Forest - RMSE train:", round(rmse_rf_train, 4),
    " | RMSE valid:", round(rmse_rf_valid, 4), "\n")

# RMSE comparison table
cmp_rmse <- data.frame(
  Model = c("Linear", "Random Forest"),
  RMSE_Train = c(rmse_lin_train, rmse_rf_train),
  RMSE_Valid = c(rmse_lin_valid, rmse_rf_valid)
)
print(cmp_rmse)

# Feature importance (Random Forest)
imp_mat <- importance(rf_model, type = 1)
imp_df  <- data.frame(Variable = rownames(imp_mat),
                      IncMSE = imp_mat[, 1], row.names = NULL) %>%
  arrange(desc(IncMSE))

print(imp_df)

# Feature importance plot
ggplot(imp_df, aes(x = reorder(Variable, IncMSE), y = IncMSE)) +
  geom_col() +
  coord_flip() +
  labs(title = "Feature Importance (%IncMSE) - Random Forest",
       x = "Variable", y = "% MSE increase when permuted") +
  theme_minimal()

# Predicted vs Observed plot (Random Forest - Validation)
ggplot(data.frame(pred = pred_rf_valid, obs = valid_df$eaux_train_ac),
       aes(x = pred, y = obs)) +
  geom_point(alpha = 0.25) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color="red") +
  labs(title = "Random Forest (2nd Fortnight): Predicted vs Observed",
       x = "Predicted (kWh)", y = "Observed (kWh)") +
  theme_minimal()

# RESULTS COMPARISON:
# - Linear Regression: RMSE train = 0.0620 | RMSE valid = 0.0586
# - Random Forest:     RMSE train = 0.1303 | RMSE valid = 0.1359
#
# Random Forest does NOT improve on the linear model: its RMSE is clearly higher
# in both training and validation. The linear model remains the most accurate
# for this problem with these variables.
#
# Generalization:
# RF has similar RMSE in train and valid (0.1303 vs 0.1359): no overfitting,
# but clear underfitting with these hyperparameters/architecture.
# The linear model achieves lower error and also generalizes well.
#
# Feature Importance (RF):
# Most important variable is event_duration, followed by:
# suma_hvac_ton_compresores, suma_ton_compresores, suma_hvac_ton_heater, etc.
# While importance ordering is useful, RF doesn't provide interpretable kW 
# coefficients. The linear model maintains direct physical interpretation.


## b) Select final model to deploy ####
# For operational and energy analysis purposes, the LINEAR MODEL is most 
# appropriate for deployment because it offers an optimal balance between:
# - Accuracy (lower RMSE)
# - Physical interpretability (coefficients in kW per component)
# - Simplicity and maintainability
#
# The Random Forest model doesn't improve RMSE and loses interpretability.
# It could be used for contrast or in purely predictive scenarios after 
# hyperparameter tuning. For operational deployment, the LINEAR MODEL is selected.
