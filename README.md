# Blackjack Optimizer

This is a simple educational Rust project to generate optimal strategies for certain varieties of Blackjack and to
evaluate expected returns in different situations.

## Scope

Some ways Blackjack can vary:

1. Does the dealer hit on soft 17?
2. Do you push if the dealer busts harder than you, or do you still lose?
3. Is there a base fee for each game played regardless of the outcome? How high is that fee?
4. Can you double after splitting? 

Some use cases:

1. See how the deck composition changes the outcome distribution.
2. See how various variations in the rules change the outcome distribution.

## Setup

This project requires a standard Rust installation to run. See [the Rust install docs](https://www.rust-lang.org/tools/install)
for an installation guide.

After that, a standard `cargo run` or `cargo run -r` from the repository root is the only way to run the project at this time.

## Contribution Guide

Just leave things in a better spot than you found them. Name things properly and put them in the right module (so far everything
is just in the root module, which is not good). Avoid pushing to main even if I haven't configured the repo to forbid it yet.