require(lme4)


run_stats_test_exp1 <- function(data_dir) {
    sentence_data <- read.csv("context_social_gender/analyses/full_sample/llama-3.1-8B/sentence_gender_association_scores.csv")
    sentence_data <- sentence_data[c("sentence_format", "feminine_score", "genderedness_score")]

    data <- read.csv(paste(data_dir, "/revision.csv", sep=""))
    data <- data[data$task_wording == "unknown gender", ]

    data["changed"] <- ifelse(data["variant_removed"] == "True", 1, 0)

    data["original_masc"] <- ifelse(data["role_noun_gender"] == "masculine", 1, 0)
    data["original_fem"] <- ifelse(data["role_noun_gender"] == "feminine", 1, 0)
    data["original_neut"] <- ifelse(data["role_noun_gender"] == "neutral", 1, 0)
    data["original_gend"] <- ifelse(data["role_noun_gender"] == "masculine" | data["role_noun_gender"] == "feminine", 1, 0)


    data <- merge(x = data, y = sentence_data, by = "sentence_format", all.x = TRUE)

    data["context_fem"] <- data["feminine_score"]
    data["context_masc"] <- -data["feminine_score"]
    data["context_gend"] <- data["genderedness_score"]
    data["context_neut"] <- -data["genderedness_score"]

    data$sentence_format <- as.factor(data$sentence_format)
    data$role_noun_set <- as.factor(data$role_noun_set)


    # train logistic regression
    model <- glmer(
        changed ~ 
        original_masc + original_fem +
        context_fem + context_neut +
        
        original_masc * context_fem +
        original_fem * context_masc +
        original_gend * context_neut +
        
        (1 | sentence_format) + (1 | role_noun_set),

        data = data, family = binomial, control = glmerControl(optimizer = "bobyqa"),)

    sink(paste(data_dir, "/regression_results_exp1.txt", sep=""))
    print(summary(model))
    sink()
    write.csv(summary(model)$coefficients, paste(data_dir, "/regression_results_exp1.csv", sep=""))
}


run_stats_test_exp2 <- function(data_dir) {
    sentence_data <- read.csv("context_social_gender/analyses/full_sample/llama-3.1-8B/sentence_gender_association_scores.csv")
    sentence_data <- sentence_data[c("sentence_format", "feminine_score", "genderedness_score")]

    data <- read.csv(paste(data_dir, "/revision.csv", sep=""))
    data <- data[data$task_wording != "unknown gender", ]

    data["changed"] <- ifelse(data["variant_removed"] == "True", 1, 0)

    data["original_masc"] <- ifelse(data["role_noun_gender"] == "masculine", 1, 0)
    data["original_fem"] <- ifelse(data["role_noun_gender"] == "feminine", 1, 0)
    data["original_neut"] <- ifelse(data["role_noun_gender"] == "neutral", 1, 0)
    data["original_gend"] <- ifelse(data["role_noun_gender"] == "masculine" | data["role_noun_gender"] == "feminine", 1, 0)

    data["prompt_gender_dec"] <- ifelse(grepl("gender declaration",data$task_wording), 1, 0)
    data["prompt_pronoun_dec"] <- ifelse(grepl("pronoun declaration",data$task_wording), 1, 0)

    data["prompt_fem"] <- ifelse(data["task_wording"] == "gender declaration woman" | data["task_wording"] == "pronoun declaration she/her" | data["task_wording"] == "pronoun usage her", 1, 0)
    data["prompt_masc"] <- ifelse(data["task_wording"] == "gender declaration man" | data["task_wording"] == "pronoun declaration he/him" | data["task_wording"] == "pronoun usage his", 1, 0)
    data["prompt_neut"] <- ifelse(data["task_wording"] == "gender declaration nonbinary" | data["task_wording"] == "pronoun declaration they/them" | data["task_wording"] == "pronoun usage their", 1, 0)


    data <- merge(x = data, y = sentence_data, by = "sentence_format", all.x = TRUE)

    data["context_fem"] <- data["feminine_score"]
    data["context_masc"] <- -data["feminine_score"]
    data["context_gend"] <- data["genderedness_score"]
    data["context_neut"] <- -data["genderedness_score"]

    data$sentence_format <- as.factor(data$sentence_format)
    data$role_noun_set <- as.factor(data$role_noun_set)


    # train logistic regression
    model <- glmer(
        changed ~ original_masc + original_fem +
        context_fem + context_neut +

        prompt_gender_dec + prompt_pronoun_dec + 
        prompt_masc + prompt_fem +

        original_masc * context_fem +
        original_fem * context_masc +
        original_gend * context_neut +
        
        original_masc * prompt_fem +
        original_fem * prompt_masc +
        original_gend * prompt_neut +
        (1 | sentence_format) + (1 | role_noun_set),
        data = data, family = binomial, control = glmerControl(optimizer = "bobyqa"),)

    sink(paste(data_dir, "/regression_results_exp2.txt", sep=""))
    print(summary(model))
    sink()
    write.csv(summary(model)$coefficients, paste(data_dir, "/regression_results_exp2.csv", sep=""))
}


models <- c("gpt-4o", "llama-3.1-8B-Instruct", "gemma-2-9b-it", "Mistral-Nemo-Instruct-2407")
for (model_name in models) {
    data_dir <- sprintf("analyses/full_revise_if_needed/%s", model_name)
    print(data_dir)

    run_stats_test_exp1(data_dir)
    run_stats_test_exp2(data_dir)
}
