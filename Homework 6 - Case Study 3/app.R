library(shiny)
library(ggplot2)
library(caret)
library(visreg)
library(bslib)

# Define UI
ui <- fluidPage(
  titlePanel("Prostate Cancer Risk Level Prediction"),
  sidebarLayout(
    sidebarPanel(
      h4("Select Predictors"),
      
      # Dynamic predictor selection based on visualization type
      uiOutput("predictorSelectionUI"),
      
      # Tab options
      hr(),
      h4("Visualization Options"),
      radioButtons("visType", "Visualization Type:",
                   choices = list("Individual Effects (Separate Plots)" = "individual", "Combined Effects (Overlay)" = "overlay", "All Effects (Grid)" = "grid"), selected = "individual"), width = 3
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Model Summary", verbatimTextOutput("modelSummary"), p("The model summary shows a linear regression model analyzing the data relationships. The model gives insight into how each data element influences the risk factors and helps make predictions.")),
        tabPanel("Prediction vs Actual", plotOutput("predictionPlot"), p("The Prediction vs Actual Risk Levels plots allow us to evaluate how well our linear regression model is performing by compairing what the model is predicting against what is actually happening in the data. The closer the points are to the line the better the model is performing meaning we would be able to have a higher degree of confidence in the model's predictions.")),
        tabPanel("Performance Metrics", verbatimTextOutput("metrics"), p("The performance metrics provide an evaluation of how well the linear regression model is performing. PRESS measures the total prediction error, RMSEP shows the average prediction accuracy, SST represents the total variation in the data, and R-squared indicates what percentage of that variation your model can explain.")),
        tabPanel("Regression Visualization", 
                 plotOutput("visregPlot", height = "600px"),
                 br(),
                 p("Note: Multiple predictors show their individual effects while controlling for other variables in the model. Regression Visualization shows the relationship between the predictor variables and the outcome (In our case, the outcome is prostate cancer risk level). This helps us identify patterns, visually assess predictors, and see how different predictors come together.
                   ")),
        tabPanel("Details of the Algorithm", verbatimTextOutput("details"), p("The purpose behind regression is to model relationships between a singular dependent variable y and independent variables X by fitting a linear equation. This information can be used to predict outcomes and give a sense of how changes in independent variables can affect the outcomes of other variables. Regression works by taking a dependent variable and an independent variable and finding the best straight line through the points. In this data set the factors such as bmi, smoker, amount of sleep, and other variables influence the likelihood that someone will develop prostate cancer. The mathematical foundation behind finding the line of best fit through the data uses the function y = mx + b where y is the dependent variable, x is the independent variable and m is the slope of the line. and b is the y intercept where the line intercepts the y axis. This form of machine learning is effective because it can quickly draw connections between variables giving the user a good insight into what is more likely to cause a specific outcome. Furthermore, linear regression can be used with more than one predictor by expanding the equation it allows each predictor to have its own coefficient. By using more predictors in some cases it will lead to better predictions and more accurate models. It is important to keep in mind in the real world there are many factors that influence the outcome.")),
        tabPanel("About", verbatimTextOutput("about"), p("This data set focuses on factors that may or may not result in a greater chance for someone to develop prostate cancer. This data set comes from Kaggle and consists of synthetic data that simulates 1000 individual health profiles focusong on potential risk factors for prostate cancer. This data set included a variety of features such as age, body mass index (BMI), smoking habits, and multiple other choices. Because of the variety of information I selected Linear Regression as my methodology for reviewing this data. This methodology has a strength when it comes to analyzing independent variables with one other dependent vairable as well as in multiple dependent variables. This allowed me to look at the data by matching common factors such as age, sleep, and BMI toghether to get a better predictability on whether someone would be at high risk for prostate cancer. Regression was able to prove a clearer view of what an individual's prostate cancer risk profile may be given multiple factors.")))
    )
  )
)

# Server logic
server <- function(input, output) {
  
  # Dynamic UI
  output$predictorSelectionUI <- renderUI({
    if (is.null(input$visType) || input$visType == "individual") {
      # Single predictor selection 
      selectInput("singlePredictor", "Choose one predictor:",
                  choices = c("age", "smoker", "bmi", "alcohol_consumption", "physical_activity_level", "mental_stress_level", "sleep_hours", "family_history", "diet_type"), selected = "age")
    } else {
      # Multiple predictor selection 
      checkboxGroupInput("predictors", "Choose predictors:", choices = c("age", "smoker", "bmi", "alcohol_consumption", "physical_activity_level", "mental_stress_level", "sleep_hours", "family_history", "diet_type"),
                         selected = c("age", "smoker", "bmi"))
    }
  })
  
  #  get current predictors
  currentPredictors <- reactive({
    if (is.null(input$visType) || input$visType == "individual") {
      req(input$singlePredictor)
      input$singlePredictor
    } else {
      req(input$predictors)
      input$predictors
    }
  })
  
  # Load and preprocess dataset
  data <- reactive({
    df <- read.csv("synthetic_prostate_cancer_risk.csv")
    
    # Convert risk_level to numeric
    df$risk_level <- ifelse(df$risk_level == "Low", 1,
                            ifelse(df$risk_level == "Medium", 2, 3))
    
    # Convert character columns to factors
    char_cols <- sapply(df, is.character)
    df[char_cols] <- lapply(df[char_cols], as.factor)
    
    df <- df[complete.cases(df), ]
    df
  })
  
  # Reactive modeling pipeline
  modelData <- reactive({
    predictors <- currentPredictors()
    req(predictors)
    
    df <- data()
    
    # Split data
    set.seed(2015)
    split <- createDataPartition(df$risk_level, p = 0.8, list = FALSE)
    train <- df[split, ]
    test <- df[-split, ]
    

    formula_text <- paste("risk_level ~", paste(predictors, collapse = " + "))
    model <- lm(as.formula(formula_text), data = train)
    

    pred <- predict(model, newdata = test, level = 0.95, interval = "confidence")
    pred_df <- data.frame(pred)
    pred_df$Reference <- test$risk_level
    

    PRESS <- sum((pred_df$Reference - pred_df$fit)^2)
    RMSEP <- sqrt(PRESS / nrow(pred_df))
    SST <- sum((pred_df$Reference - mean(pred_df$Reference))^2)
    R2 <- 1 - (PRESS / SST)
    

    if (length(predictors) == 1) {
      # for single predictor, create the reverse regression through origin
      lro_formula <- as.formula(paste(predictors[1], "~ 0 + risk_level"))
      lro <- lm(lro_formula, data = train)
    } else {
      # for multiple predictors, use the first predictor for the black line
      lro_formula <- as.formula(paste(predictors[1], "~ 0 + risk_level"))
      lro <- lm(lro_formula, data = train)
    }
    
    list(model = model, train = train, test = test, pred_df = pred_df, PRESS = PRESS, RMSEP = RMSEP, SST = SST, R2 = R2, lro = lro  )
  })
  
  # Output: Model summary
  output$modelSummary <- renderPrint({
    summary(modelData()$model)
  })
  
  # Output: Prediction vs Actual Plot
  output$predictionPlot <- renderPlot({
    df <- modelData()$pred_df
    lro <- modelData()$lro
    

    df$Risk_Category <- factor(df$Reference, 
                               levels = c(1, 2, 3),
                               labels = c("Low Risk (1)", "Medium Risk (2)", "High Risk (3)"))
    
    p2 <- ggplot(df, aes(x = Reference, y = fit, color = Risk_Category)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_errorbar(aes(ymin = lwr, ymax = upr), width = 0.2, alpha = 0.6) +
      
      # Red linear regression line
      geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "solid", 
                  show.legend = FALSE) +
      
 
      scale_color_manual(values = c("Low Risk (1)" = "#2E8B57", "Medium Risk (2)" = "#FF8C00", "High Risk (3)" = "#DC143C"),
                         name = "Actual Risk Level") +
      labs(title = "Predicted vs Actual Risk Level", subtitle = "Red line: linear regression trend | Black Dotted Line: regression through origin", x = "Actual Risk Level", y = "Predicted Risk Level") +
      theme_minimal() + theme(legend.position = "right", legend.title = element_text(size = 12, face = "bold"), legend.text = element_text(size = 10)) + coord_equal()
    
    # Add black regression line through origin 
    p2 + geom_abline(slope = lro$coefficients[1], intercept = 0, color = "black", linetype = "dashed", size = 1)
  })
  
 
  output$metrics <- renderPrint({
    cat("PRESS:", round(modelData()$PRESS, 3), "\n")
    cat("RMSEP:", round(modelData()$RMSEP, 3), "\n")
    cat("SST:", round(modelData()$SST, 3), "\n")
    cat("R-squared:", round(modelData()$R2, 4), "\n")
  })
  

  output$visregPlot <- renderPlot({
    predictors <- currentPredictors()
    req(predictors)
    
    df <- data()
    
    # build formula using all selected predictors
    formula <- as.formula(paste("risk_level ~", paste(predictors, collapse = " + ")))
    
    # Ensure all predictors exist
    if (all(predictors %in% names(df))) {
      model <- lm(formula, data = df)
      
      if (!is.null(model) && length(predictors) > 0) {
        if (input$visType == "individual") {
          # individual effects single predictor only
          visreg(model, predictors[1], main = paste("Effect of", predictors[1]))
        } else if (input$visType == "overlay") {
          if (length(predictors) <= 3) {
            visreg(model, overlay = TRUE)
            title(paste("Overlaid Effects of:", paste(predictors, collapse = ", ")))
          } else {
            visreg(model)
            title(paste("Individual Effects of:", paste(predictors, collapse = ", "), 
                        "\n(Too many for overlay)"))
          }
        } else if (input$visType == "grid") {
          # Show all effects in a grid layout
          par(mfrow = c(ceiling(length(predictors)/3), min(3, length(predictors))))
          for (pred in predictors) {
            visreg(model, pred, main = paste("Effect of", pred))
          }
          par(mfrow = c(1, 1))  # Reset layout
        }
      }
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)