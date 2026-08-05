#!/usr/bin/env Rscript
usage <- paste(
  "usage: Rscript analysis.R fit [--output-dir DIR] [--seed N]",
  "       Rscript analysis.R describe <model.rds>", sep = "\n")

args <- commandArgs(trailingOnly = TRUE)
mode <- NULL
INPUT_PATH <- NULL
OUTPUT_DIR_FLAG <- NULL
SEED_FLAG <- NULL

i <- 1
while (i <= length(args)) {
  a <- args[i]
  if (a == "--output-dir") {
    if (i == length(args)) stop("--output-dir needs a value\n", usage)
    OUTPUT_DIR_FLAG <- args[i + 1]; i <- i + 2
  } else if (a == "--seed") {
    if (i == length(args)) stop("--seed needs a value\n", usage)
    SEED_FLAG <- as.integer(args[i + 1]); i <- i + 2
  } else if (is.null(mode)) {
    mode <- a; i <- i + 1
  } else if (is.null(INPUT_PATH)) {
    INPUT_PATH <- a; i <- i + 1
  } else {
    stop("unexpected argument: ", a, "\n", usage)
  }
}

if (is.null(mode) || !(mode %in% c("fit", "describe"))) stop(usage)
if (!is.null(OUTPUT_DIR_FLAG) && mode != "fit")
  stop("--output-dir only applies to fit\n", usage)
if (mode == "fit" && !is.null(INPUT_PATH)) stop("fit takes no positional argument\n", usage)
if (mode == "describe" && is.null(INPUT_PATH))
  stop("describe needs a fitted model rds\n", usage)

if (!is.null(INPUT_PATH) && !file.exists(INPUT_PATH))
  stop("input file not found: ", INPUT_PATH)


# ============================================
# CONFIGURATION
# ============================================

INPUT_FILE          <- "data/inputfeatures.csv"
RANDOM_SEED         <- if (is.null(SEED_FLAG)) 1337 else SEED_FLAG
N_THREADS           <- max(1, parallel::detectCores() - 1)

n_clusters          <- 2
n_states            <- 5
min_sequence_length <- 4
max_sequence_length <- 22
n_restarts          <- 10
EM_MAXEVAL          <- 5000
EM_RELTOL           <- 1e-10

output_dir <- if (is.null(OUTPUT_DIR_FLAG)) "mhmm_output" else OUTPUT_DIR_FLAG

# package attach order matters
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  if (mode == "fit") library("seqHMM")
})

fit_paths <- function(dir, stem) {
  list(model   = file.path(dir, paste0(stem, ".rds")),
       csv     = file.path(dir, paste0(stem, "_cluster_assignments.csv")),
       session = file.path(dir, paste0(stem, "_session_data.rds")))
}
next_free_stem <- function(dir) {
  n <- 1
  repeat {
    stem <- sprintf("mhmm_%02d", n)
    if (!any(file.exists(unlist(fit_paths(dir, stem))))) return(stem)
    n <- n + 1
  }
}

format_duration <- function(seconds) {
  hours <- floor(seconds / 3600)
  mins <- floor((seconds %% 3600) / 60)
  secs <- round(seconds %% 60, 1)
  if (hours > 0) sprintf("%dh %dm %gs", hours, mins, secs)
  else if (mins > 0) sprintf("%dm %gs", mins, secs)
  else sprintf("%.1fs", secs)
}


# ============================================
# SHARED DATA PREP
# ============================================

prepare_sequences <- function() {
  mhmm_families <- list(
    "challenge_understanding"         = "understanding",
    "code_generation"                 = "implementation",
    "concept_guidance_non_procedural" = "understanding",
    "concept_guidance_procedural"     = "implementation",
    "confirmation"                    = "verification",
    "confusion"                       = "confusion",
    "course_platform"                 = "other",
    "debugging"                       = "verification",
    "direct_solution"                 = "outsourcing",
    "help_request"                    = "confusion",
    "non_sequitur"                    = "other",
    "observations"                    = "context",
    "pasted_context_dump"             = "context",
    "request_direction"               = "confusion",
    "social_turn"                     = "other",
    "vague_request"                   = "confusion"
  )
  data <- read.csv(INPUT_FILE)
  cat(sprintf("Loaded %d rows: %d sessions from %d users\n", nrow(data),
              length(unique(data$session_id)), length(unique(data$user_id))))

  data_mapped <- data %>%
    mutate(task_code = sapply(task_code, function(code) {
      mapped <- mhmm_families[[code]]
      if (is.null(mapped)) code else mapped
    }))

  sequences <- data_mapped %>%
    arrange(session_id, messagenum) %>%
    group_by(session_id) %>%
    summarise(
      user_id = first(user_id),
      task_sequence = list(task_code),
      original_length = n(),
      solved = first(solved),
      module_num = first(module_num),
      .groups = "drop"
    )

  # ---- filter/truncate sequences ----
  n_too_short <- sum(sequences$original_length < min_sequence_length)
  n_too_long <- sum(sequences$original_length > max_sequence_length)

  # short sessions are saved alongside the sequences in the session-data rds
  short_sessions <- sequences %>%
    filter(original_length < min_sequence_length)

  sequences <- sequences %>%
    filter(original_length >= min_sequence_length) %>%
    mutate(
      was_truncated = original_length > max_sequence_length,
      task_sequence = lapply(task_sequence, function(seq) {
        if (length(seq) > max_sequence_length) seq[1:max_sequence_length] else seq
      }),
      sequence_length = sapply(task_sequence, length)
    )

  cat(sprintf("Sequences: %d final; %d short (< %d) set aside; %d truncated at %d\n",
              nrow(sequences), n_too_short, min_sequence_length, n_too_long,
              max_sequence_length))

  list(sequences = sequences, short_sessions = short_sessions)
}

# Attach assignments and write the cluster csv + session-data rds.
write_session_outputs <- function(sequences, short_sessions, cluster_assignments,
                                  cluster_probs, output_csv, session_data_file) {
  sequences$cluster <- cluster_assignments
  for (k in 1:n_clusters) {
    sequences[[paste0("prob_c", k)]] <- cluster_probs[, k]
  }

  export_df <- sequences
  export_df$task_sequence <- vapply(
    export_df$task_sequence,
    function(x) paste(x, collapse = " "),
    character(1)
  )

  write.csv(export_df, output_csv, row.names = FALSE)
  cat("\nCluster assignments exported to:", output_csv, "\n")

  saveRDS(list(
    sequences = sequences,
    short_sessions = short_sessions
  ), session_data_file)
  cat("\nSession data saved to:", session_data_file, "\n")

  sequences
}


# ============================================
# MODE: fit
# ============================================

if (mode == "fit") {

  script_start <- Sys.time()
  if (!dir.exists(output_dir)) dir.create(output_dir)

  fit_stem <- next_free_stem(output_dir)
  paths <- fit_paths(output_dir, fit_stem)

  cat("=== MHMM FIT ===\n")
  cat("Clusters:", n_clusters, "\n")
  cat("States per cluster:", n_states, "\n")
  cat("Sequence length:", min_sequence_length, "-", max_sequence_length, "\n")
  cat("EM restarts:", n_restarts, "\n")
  cat("Output stem:", file.path(output_dir, fit_stem), "\n\n")

  prep <- prepare_sequences()
  sequences <- prep$sequences
  short_sessions <- prep$short_sessions

  # ---- build sequence object ----
  all_states <- unique(unlist(sequences$task_sequence))
  cat("\nUnique states:", length(all_states), "\n")
  cat("  ", paste(sort(all_states), collapse = ", "), "\n")

  max_len <- max(sequences$sequence_length)
  seq_matrix <- matrix(NA, nrow = nrow(sequences), ncol = max_len)
  for (i in 1:nrow(sequences)) {
    seq_len <- length(sequences$task_sequence[[i]])
    seq_matrix[i, 1:seq_len] <- sequences$task_sequence[[i]]
  }

  seq_obj <- seqdef(seq_matrix, alphabet = all_states)
  n_sequences <- nrow(seq_obj)

  # ---- fit from scratch (this mode NEVER loads a cached model) ----
  cat("\n=== FITTING MODEL ===\n")
  cat("Clusters:", n_clusters, "\n")
  cat("States per cluster:", n_states, "\n")
  cat("Sequences:", n_sequences, "\n")
  cat("EM restarts:", n_restarts, "\n")
  cat("Threads:", N_THREADS, "\n\n")

  cat("Building initial model...\n")
  fit_start <- Sys.time()

  init_model <- build_mhmm(
    observations = seq_obj,
    n_states = rep(n_states, n_clusters)
  )

  cat("Fitting model...\n")
  set.seed(RANDOM_SEED)

  fit_result <- fit_model(
    init_model,
    em_step = TRUE,
    global_step = FALSE,
    local_step = FALSE,
    threads = N_THREADS,
    control_em = list(
      restart = list(times = n_restarts),
      maxeval = EM_MAXEVAL,
      reltol = EM_RELTOL
    )
  )

  fit_end <- Sys.time()
  fit_duration <- as.numeric(difftime(fit_end, fit_start, units = "secs"))

  cat("\nFIT COMPLETE\n")
  cat("Time:", format_duration(fit_duration), "\n")
  cat("Log-likelihood:", round(logLik(fit_result$model), 2), "\n")
  cat("BIC:", round(BIC(fit_result$model), 2), "\n")
  cat("AIC:", round(AIC(fit_result$model), 2), "\n")

  final_model <- fit_result$model
  rm(fit_result, init_model)
  gc(verbose = FALSE)

  # ---- compute cluster assignments ----
  cat("\nComputing cluster assignments...\n")
  cluster_start <- Sys.time()

  if (!is.null(final_model$coefficients) && is.matrix(final_model$coefficients) &&
      ncol(final_model$coefficients) == n_clusters) {
    log_odds <- as.vector(final_model$coefficients[1, ])
    mixing_probs <- exp(log_odds) / sum(exp(log_odds))
  } else if (!is.null(final_model$prior_probs) && length(final_model$prior_probs) == n_clusters) {
    mixing_probs <- final_model$prior_probs
  } else {
    cat("  Warning: Could not extract mixing proportions, using uniform priors\n")
    mixing_probs <- rep(1/n_clusters, n_clusters)
  }

  log_mixing_probs <- log(mixing_probs)
  cluster_probs <- matrix(0, nrow = n_sequences, ncol = n_clusters)

  for (i in 1:n_sequences) {
    if (i %% 500 == 0) cat("  ", i, "/", n_sequences, "\n")

    seq_i <- seq_obj[i, , drop = FALSE]
    log_probs <- numeric(n_clusters)

    for (k in 1:n_clusters) {
      hmm_k <- build_hmm(
        observations = seq_i,
        initial_probs = final_model$initial_probs[[k]],
        transition_probs = final_model$transition_probs[[k]],
        emission_probs = final_model$emission_probs[[k]]
      )
      log_probs[k] <- log_mixing_probs[k] + logLik(hmm_k)
    }

    max_lp <- max(log_probs)
    if (is.finite(max_lp)) {
      log_probs <- log_probs - max_lp
      probs <- exp(log_probs)
      cluster_probs[i, ] <- probs / sum(probs)
    } else {
      # all clusters at -Inf log-likelihood: fall back to priors
      cluster_probs[i, ] <- mixing_probs
    }
  }

  cluster_assignments <- as.integer(apply(cluster_probs, 1, which.max))

  cluster_end <- Sys.time()
  cluster_duration <- as.numeric(difftime(cluster_end, cluster_start, units = "secs"))
  cat("Cluster assignment time:", format_duration(cluster_duration), "\n")
  cat("Cluster distribution:", paste(table(cluster_assignments), collapse = ", "), "\n")

  # ---- cluster summary ----
  cat("\n=== CLUSTER DISTRIBUTION ===\n")
  print(table(cluster_assignments))
  cat("\nProportions:\n")
  print(round(prop.table(table(cluster_assignments)), 3))

  max_probs <- apply(cluster_probs, 1, max)
  cat("\nAssignment Certainty:\n")
  cat("  Mean:", round(mean(max_probs), 3), "\n")
  cat("  Median:", round(median(max_probs), 3), "\n")

  # ---- export cluster assignments + session data ----
  sequences <- write_session_outputs(sequences, short_sessions,
                                     cluster_assignments, cluster_probs,
                                     paths$csv, paths$session)

  # ---- performance outcome analysis ----
  cat("\n=== PERFORMANCE OUTCOME ANALYSIS ===\n")

  completion_by_cluster <- NULL

  if ("solved" %in% names(sequences)) {

    completion_by_cluster <- sequences %>%
      group_by(cluster) %>%
      summarise(
        n_sessions = n(),
        n_solved = sum(solved, na.rm = TRUE),
        completion_rate = mean(solved, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        pct_of_total = n_sessions / sum(n_sessions) * 100,
        completion_pct = completion_rate * 100
      )

    cat("\nCompletion Rate by Cluster:\n")
    for (k in 1:nrow(completion_by_cluster)) {
      row <- completion_by_cluster[k, ]
      cat(sprintf("Cluster %d: %5.1f%% completion (n=%d)\n",
                  row$cluster, row$completion_pct, row$n_sessions))
    }

    cat("\n--- Naive chi-square (ignores student nesting) ---\n\n")

    contingency <- table(sequences$cluster, sequences$solved)
    chi_test <- chisq.test(contingency)
    cat(sprintf("  Naive X^2 = %.2f, df = %d, p = %.4f\n",
                chi_test$statistic, chi_test$parameter, chi_test$p.value))

    n <- sum(contingency)
    cramers_v <- sqrt(chi_test$statistic / (n * (min(dim(contingency)) - 1)))
    cat(sprintf("  Cramer's V = %.3f\n", cramers_v))

    sequences$module_factor <- relevel(factor(sequences$module_num), ref = "1")

    cat("\n--- Cluster Distribution by Module ---\n")

    cluster_by_module <- sequences %>%
      group_by(module_num, cluster) %>%
      summarise(n = n(), .groups = "drop") %>%
      pivot_wider(names_from = cluster, values_from = n, values_fill = 0,
                  names_prefix = "C")

    cat("\nCounts (sessions per cluster within each module):\n")
    print(cluster_by_module, width = Inf)

    cluster_by_module_pct <- sequences %>%
      group_by(module_num, cluster) %>%
      summarise(n = n(), .groups = "drop") %>%
      group_by(module_num) %>%
      mutate(pct = n / sum(n) * 100) %>%
      select(-n) %>%
      pivot_wider(names_from = cluster, values_from = pct, values_fill = 0,
                  names_prefix = "C")

    cat("\nRow percentages (cluster distribution within each module):\n")
    print(cluster_by_module_pct, width = Inf)

    cluster_module_table <- table(sequences$cluster, sequences$module_num)
    chi_cluster_module <- chisq.test(cluster_module_table)
    cat(sprintf("\nChi-squared test (cluster x module independence):\n"))
    cat(sprintf("  X^2 = %.2f, df = %d, p = %.4f\n",
                chi_cluster_module$statistic, chi_cluster_module$parameter, chi_cluster_module$p.value))

    n_total_cm <- sum(cluster_module_table)
    cramers_v_cm <- sqrt(chi_cluster_module$statistic / (n_total_cm * (min(dim(cluster_module_table)) - 1)))
    cat(sprintf("  Cramer's V = %.3f\n", cramers_v_cm))

  } else {
    cat("\nWARNING: 'solved' variable not found in data.\n")
  }

  # ---- save fit outputs (model and assignments) ----
  cat("\nSaving model and assignments...\n")
  saveRDS(list(
    model = final_model,
    cluster_assignments = cluster_assignments,
    cluster_probs = cluster_probs,
    mixing_probs = mixing_probs,
    n_clusters = n_clusters,
    n_states = n_states,
    n_sequences = n_sequences,
    min_length = min_sequence_length,
    max_length = max_sequence_length,
    truncate = TRUE,
    n_restarts = n_restarts,
    em_maxeval = EM_MAXEVAL,
    em_reltol = EM_RELTOL,
    seed = RANDOM_SEED,
    use_covariates = FALSE,
    logLik = logLik(final_model),
    BIC = BIC(final_model),
    AIC = AIC(final_model),
    fit_time = fit_duration
  ), paths$model)
  cat("Model saved to:", paths$model, "\n")

  total_time <- as.numeric(difftime(Sys.time(), script_start, units = "secs"))
  cat("\nTotal time:", format_duration(total_time), "\n")
}


# ============================================
# MODE: describe
# ============================================
if (mode == "describe") {
  cached <- readRDS(INPUT_PATH)
  m <- cached$model
  cat("=== FITTED MODEL ===\n")
  cat("MHMM:", cached$n_clusters, "clusters x", cached$n_states, "states;",
      cached$n_sequences, "sequences\n")
  cat(sprintf("logLik = %.1f   BIC = %.1f   AIC = %.1f\n",
              as.numeric(cached$logLik), cached$BIC, cached$AIC))
  cat("Cluster sizes:", paste(table(cached$cluster_assignments), collapse = " / "), "\n")
  mp <- apply(cached$cluster_probs, 1, max)
  cat(sprintf("Assignment certainty: mean %.3f, median %.3f, %.1f%% of conversations >= 0.70\n",
              mean(mp), median(mp), 100 * mean(mp >= 0.7)))
  cat("\n=== INITIAL / TRANSITION / EMISSION MATRICES ===\n")
  for (k in seq_along(m$initial_probs)) {
    cat(sprintf("\n--- Cluster %d ---\n", k))
    cat("initial state probabilities:\n")
    print(round(m$initial_probs[[k]], 2))
    cat("\ntransition probabilities (row state -> column state):\n")
    print(round(m$transition_probs[[k]], 2))
    cat("\nemission probabilities (state x query family):\n")
    print(round(m$emission_probs[[k]], 2))
  }
}
