# Lab 4: Regression and Regularization

## Purpose

This lab explores regression techniques using the Diabetes dataset from `sklearn.datasets`. The analysis compares simple linear regression, multiple linear regression, polynomial regression, Ridge regression, and Lasso regression.

The notebook uses a reproducible 80/20 train-test split with `random_state=42`. 

## Findings

The scikit-learn Diabetes dataset had 442 samples and 10 features. Its feature values are already standardized, so I used them in their provided form. The target values ranged from 25 to 346. I checked the features and target for missing values and filtered out any incomplete rows; because none were missing, all 442 rows stayed in the analysis.

The simple linear regression selected `bmi` as the independent variable because it had the strongest correlation with the target. Its RMSE was `63.7325` and its R² was `0.2334`. This means BMI by itself was useful, but it did not explain very much of the target variation.

The multiple linear regression used all 10 features and performed better. It had an RMSE of `53.8534` and an R² of `0.4526`. The lower error compared to the simple model shows that the other health measurements added useful information.

For polynomial regression, degree 1 had the best test results with an RMSE of `53.8534`. Degree 2 was a little worse with an RMSE of `55.6420`. Degree 3 had a training R² of `0.8773`, but its test RMSE increased to `287.1342` and its test R² became `-14.5613`. This was a clear sign of overfitting because the model fit the training data very well but performed badly on new data.

Regularization gave a small improvement. I selected alpha using five-fold cross-validation on the training data, then evaluated each selected model on the held-out test set. The best Ridge model used alpha `31.6228`, with a test RMSE of `53.4933` and an R² of `0.4599`. The best Lasso model used alpha `1`, with a test RMSE of `53.1467` and an R² of `0.4669`. Lasso was the best primary model in this comparison. The cross-validation results show that very large alpha values hurt performance, especially for Lasso, whose mean CV RMSE reached `77.9816` at alpha `100`.

Ridge kept all of the features but made the coefficients smaller. Lasso reduced some coefficients more strongly and set the `s2` coefficient to `0`, which means it removed that feature from the model. The biggest positive Lasso coefficients were for `bmi` and `s5`. These coefficients show model relationships, but they should not be treated as proof of medical causation.

Overall, the Lasso model with alpha `1` performed the best on this seeded train-test split. Its improvement over multiple linear regression was modest: test RMSE decreased from `53.8534` to `53.1467`, and R² increased from `0.4526` to `0.4669`. The results also show that a more complicated model is not always better, especially when the higher-degree polynomial model starts fitting noise. Alpha selection used cross-validation on training data, leaving the test set for final evaluation.

## Challenges faced

- **Polynomial model errors or very large values:** I ran into a problem when increasing the polynomial degree. Degree 3 fit the training data much better, but the test errors became very large because the model was overfitting. I decided not to use a higher degree since it could make the predictions even more unstable.
- **Lasso convergence warning:** I also had to be careful with Lasso when using small alpha values because it can take many iterations to converge. I used `max_iter=100000` in the notebook so the model had more time to finish. If the warning still happens, the iteration limit can be increased again.


## How to Run

1. Open `nmetha_lab4.ipynb` in Jupyter Notebook or VS Code.
2. Select a Python environment with `numpy`, `pandas`, `matplotlib`, and `scikit-learn` installed.
3. Run the cells from top to bottom using a fresh kernel.
