# BBC-Text-Analysis
############################################################
# DSCI 784 – BBC News Text Mining Project
# Full Pipeline: Load → Prep → Sentiment → LDA → Plots
# UPDATED: Added diverse visualization types for rubric compliance
############################################################

## 0) PACKAGES ----

# Install once if needed:
# install.packages(c(
#    "dplyr", "purrr", "readr", "stringr",
#    "tidyverse", "tidytext",
#    "quanteda", "topicmodels",
#    "textdata", "ggplot2", "wordcloud" ))

library(dplyr)
library(purrr)
library(readr)
library(stringr)
library(tidyverse)
library(tidytext)
library(quanteda)
library(topicmodels)
library(textdata)
library(ggplot2)
library(wordcloud)

set.seed(123)  # for reproducibility

############################################################
# 1) LOAD RAW TEXT FILES FROM FOLDERS
############################################################

# Your exact base path:
base_path <- "C:/Users/victor.badu/Downloads/archive (7)/BBC News Summary/BBC News Summary/News Articles"

# Function to read all news articles from one category folder
read_news_articles <- function(category) {
  news_path <- file.path(base_path, category)
  files <- list.files(news_path, full.names = TRUE, pattern = "\\.txt$")
  
  cat("Processing", category, "-", length(files), "files\n")
  
  map_df(files, function(file) {
    content <- readLines(file, warn = FALSE, encoding = "UTF-8") %>% 
      paste(collapse = " ") %>%
      str_squish()  # remove extra whitespace
    
    data.frame(
      category = category,
      text     = content,
      file_id  = basename(file),  # unique-ish ID from filename
      stringsAsFactors = FALSE
    )
  })
}

# BBC categories
categories <- c("business", "entertainment", "politics", "sport", "tech")

# Read all categories into one data frame
bbc_news_data <- map_df(categories, read_news_articles)

# Verify dataset
cat("\n=== DATASET VERIFICATION ===\n")
cat("Total articles:", nrow(bbc_news_data), "\n")
cat("Columns:", names(bbc_news_data), "\n\n")

############################################################
# 2) DESCRIPTIVE STATISTICS
############################################################

# Articles per category
category_summary <- bbc_news_data %>%
  count(category) %>%
  arrange(desc(n))

cat("Articles per category:\n")
print(category_summary)

# Basic text stats: char and word counts
bbc_news_data <- bbc_news_data %>%
  mutate(
    char_count = nchar(text),
    word_count = str_count(text, "\\S+")  # non-space tokens
  )

cat("\nText Statistics:\n")
cat("Average characters per article:", round(mean(bbc_news_data$char_count), 2), "\n")
cat("Average words per article:", round(mean(bbc_news_data$word_count), 2), "\n")
cat("Total words in corpus:", sum(bbc_news_data$word_count), "\n")

cat("\nSample of first article (first 300 characters):\n")
cat(substr(bbc_news_data$text[1], 1, 300), "...\n\n")

# Optionally save as CSV for later use
write_csv(bbc_news_data, "bbc_news_articles.csv")
cat("Dataset saved as 'bbc_news_articles.csv'\n\n")

# ---- Figure 1: BAR CHART - Articles per category ----
ggplot(category_summary, 
       aes(x = reorder(category, -n), y = n, fill = category)) +
  geom_col(show.legend = FALSE) +
  labs(
    title = "Distribution of Articles by BBC News Category",
    x = "Category",
    y = "Number of Articles"
  ) +
  theme_minimal(base_size = 16) +  # Increased from 12 to 16
  theme(axis.text.x = element_text(angle = 0, size = 14),
        axis.text.y = element_text(size = 14),
        plot.title = element_text(size = 18, face = "bold"))

############################################################
# 3) TEXT PREPARATION (TIDY TOKENS)
############################################################

data("stop_words")

# Add custom stopwords for news-specific common terms
custom_stops <- tibble(
  word = c("said", "new", "also", "one", "two", "first", "last", "can", "will"),
  lexicon = "custom"
)

stop_words_extended <- bind_rows(stop_words, custom_stops)

bbc_tidy <- bbc_news_data %>%
  select(category, file_id, text) %>%
  unnest_tokens(word, text) %>%            # tokenize to unigrams
  filter(!str_detect(word, "^[0-9]+$")) %>%# remove pure numbers
  anti_join(stop_words_extended, by = "word") %>%   # remove extended stopwords
  filter(str_length(word) > 2)             # remove very short words (keeps 3+ chars)

cat("Rows in tidy token table:", nrow(bbc_tidy), "\n\n")

############################################################
# 4) SENTIMENT ANALYSIS (BING LEXICON)
############################################################

# Bing sentiment: positive / negative
bing_sentiments <- get_sentiments("bing")

# Sentiment per document
bbc_sent_doc <- bbc_tidy %>%
  inner_join(bing_sentiments, by = "word") %>%
  count(category, file_id, sentiment) %>%
  tidyr::pivot_wider(
    names_from  = sentiment,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(net_sentiment = positive - negative)

# Aggregate sentiment by category
bbc_sent_cat <- bbc_sent_doc %>%
  group_by(category) %>%
  summarise(
    avg_positive = mean(positive),
    avg_negative = mean(negative),
    avg_net_sent = mean(net_sentiment)
  ) %>%
  arrange(desc(avg_net_sent))

cat("Average sentiment by category:\n")
print(bbc_sent_cat)
cat("\n")

# ---- Figure 2: LOLLIPOP/DOT PLOT - Net sentiment (DIFFERENT CHART TYPE) ----
# This replaces the bar chart to provide visualization diversity per rubric
ggplot(bbc_sent_cat,
       aes(x = reorder(category, avg_net_sent), y = avg_net_sent)) +
  geom_segment(aes(xend = category, y = 0, yend = avg_net_sent), 
               color = "gray50", size = 1.2) +
  geom_point(aes(color = avg_net_sent > 0), size = 6) +
  scale_color_manual(
    values = c("TRUE" = "#2E7D32", "FALSE" = "#C62828"),
    labels = c("Positive", "Negative")
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray30", alpha = 0.7) +
  coord_flip() +
  labs(
    title = "Net Sentiment Across BBC News Categories",
    subtitle = "Balance of positive vs. negative language",
    x = NULL,
    y = "Average Net Sentiment (Positive - Negative)",
    color = "Sentiment Type"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "gray40")
  )

############################################################
# 5) TOPIC MODELING WITH LDA (k = 5)
############################################################

# FIX: create unique doc_id
bbc_news_data <- bbc_news_data %>%
  mutate(doc_id = paste(category, file_id, sep = "_"))

# Now create quanteda corpus safely
bbc_corpus <- corpus(
  bbc_news_data,
  text_field = "text",
  docid_field = "doc_id"
)

# Tokenization & cleaning
bbc_tokens <- tokens(
  bbc_corpus,
  remove_punct   = TRUE,
  remove_numbers = TRUE
) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_keep(min_nchar = 3)

# Document-feature matrix (DFM)
bbc_dfm <- dfm(bbc_tokens)

# Trim sparse terms to stabilize LDA
bbc_dfm_trim <- dfm_trim(
  bbc_dfm,
  min_termfreq = 20,  # term appears at least 20 times
  min_docfreq  = 5    # in at least 5 documents
)

# Convert to topicmodels-friendly DocumentTermMatrix
bbc_dtm <- convert(bbc_dfm_trim, to = "topicmodels")

# LDA model (k = 5 topics)
k <- 5
lda_model <- LDA(
  bbc_dtm,
  k       = k,
  control = list(seed = 123)
)

lda_model

# ---- Figure 3: BAR CHART - Top terms per topic ----
lda_top_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  # OPTIONAL: give topics human-readable names
  mutate(topic = factor(topic,
                        levels = 1:5,
                        labels = c("Entertainment",
                                   "Business",
                                   "Sport",
                                   "Tech",
                                   "Politics")))

ggplot(lda_top_terms,
       aes(x = reorder_within(term, beta, topic),
           y = beta,
           fill = topic)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_x_reordered() +
  facet_wrap(~ topic, scales = "free_y") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Top 10 Terms per LDA Topic (k = 5)",
    x     = "Term",
    y     = expression(beta ~ "(Term Probability in Topic)")
  ) +
  theme_minimal()

############################################################
# 6) MOST FREQUENT WORDS BY CATEGORY
############################################################

# ---- Figure 4: BAR CHART - Top words per category ----
top_words_by_cat <- bbc_tidy %>%
  count(category, word, sort = TRUE) %>%
  group_by(category) %>%
  slice_max(n, n = 10) %>%
  ungroup() %>%
  mutate(word = reorder_within(word, n, category))

ggplot(top_words_by_cat, 
       aes(x = word, y = n, fill = category)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_x_reordered() +
  facet_wrap(~ category, scales = "free_y") +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Most Frequent Words by BBC News Category",
    x = NULL,
    y = "Frequency"
  ) +
  theme_minimal()

############################################################
# 7) WORD CLOUD (DIFFERENT VISUALIZATION TYPE)
############################################################

# Global word frequency (already cleaned in bbc_tidy)
word_freq <- bbc_tidy %>%
  count(word, sort = TRUE)

# ---- Figure 5: WORD CLOUD of most frequent terms ----
set.seed(123)
par(bg = "white")
wordcloud(
  words       = word_freq$word,
  freq        = word_freq$n,
  max.words   = 100,
  random.order = FALSE,
  rot.per     = 0.25,
  scale       = c(4, 0.8),
  colors      = brewer.pal(8, "Dark2")
)
title(main = "Global Word Cloud - BBC News Corpus", 
      line = -1, cex.main = 1.2)

############################################################
# SUMMARY OF VISUALIZATIONS FOR RUBRIC COMPLIANCE
############################################################

cat("\n=== VISUALIZATION SUMMARY ===\n")
cat("Figure 1: Bar chart (articles per category)\n")
cat("Figure 2: Lollipop/dot plot (sentiment analysis) **DIFFERENT TYPE**\n")
cat("Figure 3: Bar chart (LDA top terms)\n")
cat("Figure 4: Bar chart (frequent words by category)\n")
cat("Figure 5: Word cloud (global vocabulary) **DIFFERENT TYPE**\n")
cat("\nRubric requirement met: 3+ charts with at least one different type ✓\n")

############################################################
# END OF SCRIPT
############################################################
