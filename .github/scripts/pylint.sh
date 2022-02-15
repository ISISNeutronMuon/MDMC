#!/bin/bash

# This script lints a pull request, prints new linting notes that
# aren't in master, and then passes or fails based on score.

pylint MDMC > pylint.txt
# get differences report and score for both reports
head -n -2 pylint_cron.txt > pylint_cron_notes.txt
export CRON_SCORE=$(tail -n -2 pylint_cron.txt | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p')

head -n -2 pylint_pr.txt > pylint_pr_notes.txt
export PR_SCORE=$(tail -n -2 pylint_pr.txt | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p')

# print report if any differences
echo "Linting issues in this PR's new code";
diff pylint_cron_notes.txt pylint_pr_notes.txt

echo "This PR's pylint score:"; echo "$PR_SCORE";
echo "Master pylint score:"; echo "$CRON_SCORE";
# calculate if score is lower than master and exit accordingly
awk 'BEGIN {
    pr_score = ENVIRON["PR_SCORE"]
    cron_score = ENVIRON["CRON_SCORE"]
    exit pr_score < cron_score}' || echo "Your score is lower than master - linting failed." && exit 1
echo "Your score is higher than (or equal to) master - linting passed!" && exit 0
