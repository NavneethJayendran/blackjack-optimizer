use itertools::izip;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::io::BufRead;
use std::vec;
use thiserror::Error;

pub struct BlackjackDealerPredictiveModel {
    // map from starting card value (Ace is index 0) to instances of given final dealer hand values
    hand_from_starting_card_simulated_counts: [[u64; 35]; 10],
    row_totals: [u64; 10],
    deck: Vec<Card>,
    rng: Box<dyn RngCore>,
}

#[derive(Clone, Copy, Default)]
pub struct BlackjackHand {
    number_of_cards: u32,
    number_of_aces: u32,
    non_ace_value: u32,
}

impl BlackjackHand {
    fn add_card(&mut self, card: CardValue) -> BlackjackHand {
        self.number_of_cards += 1;
        if let Some(value) = card.value() {
            self.non_ace_value += value;
        } else {
            self.number_of_aces += 1;
        }
        self.clone()
    }

    fn best_value(&self) -> u32 {
        let minimum_hand_value = self.non_ace_value + 1 * self.number_of_aces;
        let remainder_to_21 = max(0, 21 - (minimum_hand_value as i32)) as u32;
        let best_card_promotions_count = min(remainder_to_21 / 10, self.number_of_aces);
        minimum_hand_value + 10 * best_card_promotions_count
    }
}

pub struct BlackjackDealerHandModelOptions {
    deck: Option<Vec<Card>>,
}

fn standard_deck(copies: u32) -> Vec<Card> {
    let mut out = vec![];
    for _ in 0..copies {
        use Card::*;
        for card in [
            Ace, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King,
        ] {
            for _ in 0..4 {
                out.push(card)
            }
        }
    }
    out
}

struct BlackjackDealerHandView<'a, 'b> {
    rng: &'a mut Box<dyn RngCore>,
    deck: &'b mut Vec<Card>,
    deck_index: usize,
    hand: BlackjackHand,
}

impl<'a, 'b> BlackjackDealerHandView<'a, 'b> {
    pub fn new(
        rng: &'a mut Box<dyn RngCore>,
        deck: &'b mut Vec<Card>,
    ) -> BlackjackDealerHandView<'a, 'b> {
        BlackjackDealerHandView {
            rng,
            deck,
            deck_index: 0,
            hand: BlackjackHand::default(),
        }
    }

    /// Reset the deck to having no cards drawn and partially shuffle it to remove all correlation between the
    /// previous draw sequence and the next draw sequence. This only swaps as many elements as were drawn in
    /// the previous draw sequence.
    pub fn reset(&mut self) {
        for offset in 0..self.deck_index {
            let swap_index = self.rng.gen_range(offset..self.deck.len());
            self.deck.swap(offset, swap_index);
        }
        self.deck_index = 0;
        self.hand = BlackjackHand::default();
    }

    pub fn draw(&mut self) -> Card {
        let card = self.deck[self.deck_index];
        self.deck_index += 1;
        self.hand.add_card(card.into());
        card
    }

    pub fn did_draw_ace(&self) -> bool {
        self.hand.number_of_aces > 0
    }

    pub fn best_hand_value(&self) -> u32 {
        self.hand.best_value()
    }
}

impl BlackjackDealerPredictiveModel {
    pub fn new(options: &BlackjackDealerHandModelOptions) -> Self {
        let deck = if let Some(deck) = &options.deck {
            deck.clone()
        } else {
            standard_deck(1)
        };
        let mut out = BlackjackDealerPredictiveModel {
            deck: deck,
            rng: Box::new(rand::thread_rng()),
            hand_from_starting_card_simulated_counts: [[0; 35]; 10],
            row_totals: [0; 10],
        };
        out.shuffle();
        out
    }

    pub fn relative_stderrs(&mut self) -> Vec<Vec<f64>> {
        self.hand_from_starting_card_simulated_counts
            .into_iter()
            .map(|row| {
                row.iter()
                    .map(|x| {
                        if *x > 1 {
                            1.0 / ((*x - 1) as f64).sqrt()
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<f64>>()
            })
            .collect()
    }

    pub fn probabilities(&self) -> Vec<Vec<f64>> {
        self.hand_from_starting_card_simulated_counts
            .into_iter()
            .enumerate()
            .map(|(row_index, row)| {
                row.iter()
                    .map(|x| (*x as f64) / (self.row_totals[row_index] as f64))
                    .collect::<Vec<f64>>()
            })
            .collect()
    }

    pub fn final_probability(&self, start_card: Card, final_best_value: u32) -> f64 {
        if final_best_value >= 35 {
            return 0.0;
        }
        let final_hand = final_best_value as usize;
        let row_index: usize = match start_card.value() {
            None => 0,
            Some(num) => (num as usize) - 1,
        };
        (self.hand_from_starting_card_simulated_counts[row_index][final_hand] as f64)
            / (self.row_totals[row_index] as f64)
    }

    pub fn simulate(&mut self, iterations: u64) {
        let mut hand = BlackjackDealerHandView::new(&mut self.rng, &mut self.deck);
        for _ in 0..iterations {
            let card = hand.draw();
            let row_index = card.value().map_or(0, |v| v - 1) as usize;
            loop {
                let best_hand_value = hand.best_hand_value();
                if best_hand_value > 17 || best_hand_value == 17 && !hand.did_draw_ace() {
                    self.hand_from_starting_card_simulated_counts[row_index]
                        [best_hand_value as usize] += 1;
                    self.row_totals[row_index] += 1;
                    break;
                }
                _ = hand.draw();
            }
            hand.reset();
        }
    }

    fn shuffle(&mut self) {
        for offset in 0..self.deck.len() {
            let swap_index = self.rng.gen_range(offset..self.deck.len());
            self.deck.swap(offset, swap_index);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum BlackjackMove {
    Stay,
    Hit,
    Double,
    Optimal,
}

pub struct PlayerOptimizer<'a> {
    dealer_model: &'a BlackjackDealerPredictiveModel,
    base_fee_fraction: f64,
    dual_bust_protection: bool,
    cards: CompressedDeck,
}

impl<'a> PlayerOptimizer<'a> {
    fn new(
        dealer_model: &'a BlackjackDealerPredictiveModel,
        deck: &Vec<Card>,
        base_fee_fraction: f64,
        dual_bust_protection: bool,
    ) -> PlayerOptimizer<'a> {
        PlayerOptimizer {
            dealer_model,
            base_fee_fraction,
            dual_bust_protection,
            cards: CompressedDeck::new(deck.iter().map(|card| CardValue::from(*card))),
        }
    }

    fn best_move(
        &mut self,
        your_cards: &Vec<Card>,
        dealer_card: Card,
    ) -> Result<Vec<(BlackjackMove, f64)>, BasicError> {
        use BlackjackMove::*;
        let mut your_hand = BlackjackHand::default();
        if self.cards.remove(dealer_card.into()) <= 0.0 {
            return Err(BasicError::NotEnoughCards(format!(
                "Removed more {:?} cards than present in the deck",
                dealer_card
            )));
        }
        for card in your_cards {
            if self.cards.remove((*card).into()) <= 0.0 {
                return Err(BasicError::NotEnoughCards(format!(
                    "Removed more {:?} cards than present in the deck",
                    dealer_card
                )));
            }
            your_hand.add_card((*card).into());
        }
        let mut out: Vec<(BlackjackMove, f64)> = [Stay, Hit, Double]
            .iter()
            .map(|mv| self.expected_return(your_hand, dealer_card, *mv))
            .filter(|(_, expected_return)| *expected_return > f64::NEG_INFINITY)
            .collect();
        out.sort_by(|(_, a), (_, b)| a.total_cmp(b).reverse());

        self.cards.add(dealer_card.into());
        for card in your_cards {
            self.cards.add((*card).into());
        }
        Ok(out)
    }

    fn expected_return(
        &mut self,
        your_hand: BlackjackHand,
        dealer_card: Card,
        bj_move: BlackjackMove,
    ) -> (BlackjackMove, f64) {
        let your_best_value = your_hand.best_value();
        if your_best_value > 21 {
            return if your_hand.number_of_cards <= 3 && self.dual_bust_protection {
                let odds_dealer_better_hand = (0..=your_best_value)
                    .map(|finalh| self.dealer_model.final_probability(dealer_card, finalh))
                    .sum::<f64>();
                (BlackjackMove::Stay, -self.base_fee_fraction - odds_dealer_better_hand)
            } else {
                (BlackjackMove::Stay, -self.base_fee_fraction - 1.0)
            }
        }
        let recurse = |selfv: &mut Self, your_hand, bj_move| {
            selfv.expected_return(your_hand, dealer_card, bj_move)
        };
        match bj_move {
            BlackjackMove::Optimal => *(vec![
                recurse(self, your_hand, BlackjackMove::Stay),
                recurse(self, your_hand, BlackjackMove::Hit),
                recurse(self, your_hand, BlackjackMove::Double)
            ]
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1)))
            .unwrap(),
            BlackjackMove::Hit => {
                let mut total = 0.0;
                for card_value in CardValue::VALUES {
                    let prob = self.cards.remove(card_value);
                    total += prob
                        * recurse(
                            self,
                            your_hand.clone().add_card(card_value),
                            BlackjackMove::Optimal,
                        )
                        .1;
                    self.cards.add(card_value);
                }
                (bj_move, total)
            }
            BlackjackMove::Stay => {
                let can_stay = your_hand.number_of_cards >= 2;
                if !can_stay {
                    return (BlackjackMove::Stay, f64::NEG_INFINITY);
                }
                let natural_blackjack = your_hand.number_of_cards == 2 && your_best_value == 21;
                let payout_ratio: f64 = if natural_blackjack { 1.5 } else {1.0};
                let prob_loss: f64 = (your_best_value + 1..=21)
                    .map(|final_dealer_best_value| {
                        self.dealer_model
                            .final_probability(dealer_card, final_dealer_best_value)
                    })
                    .sum();
                let prob_tie = self
                    .dealer_model
                    .final_probability(dealer_card, your_best_value);
                let prob_win = 1.0 - prob_tie - prob_loss;
                (bj_move, payout_ratio * prob_win - prob_loss - self.base_fee_fraction)
            }
            BlackjackMove::Double => {
                let can_double = your_hand.number_of_cards == 2;
                if !can_double {
                    return (BlackjackMove::Double, f64::NEG_INFINITY);
                }
                let mut total = 0.0;
                for card_value in CardValue::VALUES {
                    let prob = self.cards.remove(card_value);
                    total += prob
                        * (2.0
                            * recurse(
                                self,
                                your_hand.clone().add_card(card_value),
                                BlackjackMove::Stay,
                            )
                            .1);
                    self.cards.add(card_value);
                }
                (bj_move, total + self.base_fee_fraction)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Card {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
}

impl Card {
    fn value(&self) -> Option<u32> {
        CardValue::from(*self).value()
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum CardValue {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
}

impl CardValue {
    const VALUES: [Self; 10] = [
        CardValue::Ace,
        CardValue::Two,
        CardValue::Three,
        CardValue::Four,
        CardValue::Five,
        CardValue::Six,
        CardValue::Seven,
        CardValue::Eight,
        CardValue::Nine,
        CardValue::Ten,
    ];

    fn value(&self) -> Option<u32> {
        use CardValue::*;
        match *self {
            Ace => None, // Ace is not single-valued
            Two => Some(2),
            Three => Some(3),
            Four => Some(4),
            Five => Some(5),
            Six => Some(6),
            Seven => Some(7),
            Eight => Some(8),
            Nine => Some(9),
            Ten => Some(10),
        }
    }
}

impl From<Card> for CardValue {
    fn from(card: Card) -> Self {
        match card {
            Card::Ace => CardValue::Ace,
            Card::Two => CardValue::Two,
            Card::Three => CardValue::Three,
            Card::Four => CardValue::Four,
            Card::Five => CardValue::Five,
            Card::Six => CardValue::Six,
            Card::Seven => CardValue::Seven,
            Card::Eight => CardValue::Eight,
            Card::Nine => CardValue::Nine,
            Card::Ten | Card::Jack | Card::Queen | Card::King => CardValue::Ten,
        }
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BasicError {
    #[error("Not enough cards: {0}")]
    NotEnoughCards(String),
}

struct CompressedDeck {
    total: u64,
    counts: HashMap<CardValue, u64>,
}

impl CompressedDeck {
    fn new<CardsT: Iterator<Item = CardValue>>(cards: CardsT) -> Self {
        let mut out = CompressedDeck {
            total: 0,
            counts: HashMap::new(),
        };
        for card in cards {
            out.add(card);
        }
        out
    }

    fn prob(&self, card_value: CardValue) -> f64 {
        (*self.counts.get(&card_value).unwrap_or(&0) as f64) / (self.total as f64)
    }

    fn add(&mut self, card_value: CardValue) {
        *self.counts.entry(card_value.clone()).or_insert(0) += 1;
        self.total += 1;
    }

    fn remove(&mut self, card_value: CardValue) -> f64 {
        match self.counts.entry(card_value).or_insert(0) {
            &mut 0 => 0.0,
            amount => {
                let out = (*amount as f64) / (self.total as f64);
                self.total -= 1;
                *amount -= 1;
                out
            }
        }
    }
}

fn main_wrapper() -> Option<()> {
    let stdin = std::io::stdin();
    let mut stdin_iterator = stdin.lock().lines();
    let mut read_line = move || -> Option<String> {
        let line = stdin_iterator.next();
        match line {
            None => {
                println!("No more input. Terminating.");
                None
            }
            Some(line) => Some(line.expect("Failed to read line. Terminating.")),
        }
    };

    let mut deck = standard_deck(8);
    let mut rng = rand::thread_rng();
    for offset in 0..deck.len() {
        let swap_index = rng.gen_range(offset..deck.len());
        deck.swap(offset, swap_index);
    }
    deck.shrink_to(deck.len() / 2);
    let mut model = BlackjackDealerPredictiveModel::new(&BlackjackDealerHandModelOptions {
        deck: Some(deck.clone()),
    });
    let iterations = loop {
        println!("Please enter a number of iterations to train the simulator: ");
        match read_line()?.parse::<u64>() {
            Ok(amount) => break amount,
            Err(_) => println!("Failed to parse iterations. Try again."),
        }
    };
    model.simulate(iterations);
    let probabilities = model.probabilities();
    let relative_stderrs = model.relative_stderrs();
    for i in 0..10 {
        let counts: Vec<String> = izip!(probabilities[i].iter(), relative_stderrs[i].iter())
            .enumerate()
            .filter(|(_, (prob, _))| **prob > 0.0)
            .map(|(idx, (prob, rel_stderr))| {
                format!(
                    "{}:{:.5}+-{:.5}",
                    idx,
                    *prob as f64,
                    (*prob * (1.0 - *prob)).sqrt() * rel_stderr
                )
            })
            .collect();
        println!("{}: {:?}", i + 1, counts);
    }
    let mut optimizer = PlayerOptimizer::new(&model, &deck, 0.0 / 15.0, true);

    loop {
        let dealer_card = loop {
            println!("Enter dealer card:");
            let attempt: Result<Card, _> = serde_json::from_str(&format!("\"{}\"", read_line()?));
            match attempt {
                Ok(card) => break card,
                Err(_) => println!("Failed to parse card. Try again."),
            }
        };

        let mut your_cards = vec![];
        println!("Keep entering cards for you to draw from the deck, or just hit enter to run the optimizer.");
        loop {
            let input = read_line()?;
            if input.is_empty() {
                break;
            }
            let attempt: Result<Card, _> = serde_json::from_str(&format!("\"{input}\""));
            match attempt {
                Ok(card) => your_cards.push(card),
                Err(_) => println!("Failed to parse card. Try again."),
            }
        }

        match optimizer.best_move(&your_cards, dealer_card) {
            Ok(mut move_values) => {
                println!(
                    "Your best move is {:?}, which has an expected return of {:.3}",
                    move_values[0].0, move_values[0].1
                );
                for (mv, ret) in &move_values[1..] {
                    println!("{:?} has an expected return of {:.3}", *mv, *ret)
                }
            }
            Err(e) => {
                println!("{e}");
            }
        }
        println!("");
    }
}

fn main() {
    main_wrapper();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blackjack_hand() {
        let mut hand = BlackjackHand::default();
        assert_eq!(0, hand.best_value());
        hand.add_card(Card::Ace.into());
        assert_eq!(11, hand.best_value());
        hand.add_card(Card::Five.into());
        assert_eq!(16, hand.best_value());
        hand.add_card(Card::Ace.into());
        assert_eq!(17, hand.best_value());
        hand.add_card(Card::Ten.into());
        assert_eq!(17, hand.best_value());
        hand.add_card(Card::Four.into());
        assert_eq!(21, hand.best_value());
        hand.add_card(Card::Six.into());
        assert_eq!(27, hand.best_value());
    }

    #[test]
    fn test_blackjack_hand_view() {
        let mut rng: Box<dyn RngCore> = Box::new(rand::rngs::mock::StepRng::new(0, 1));
        let mut deck = vec![Card::Ace, Card::Two, Card::Three, Card::Four];
        let mut hand_view = BlackjackDealerHandView::new(&mut rng, &mut deck);
        assert_eq!(Card::Ace, hand_view.draw());
        assert_eq!(Card::Two, hand_view.draw());
        assert_eq!(Card::Three, hand_view.draw());
        assert_eq!(Card::Ace, deck[0]);
        assert_eq!(Card::Two, deck[1]);
        assert_eq!(Card::Three, deck[2]);
        assert_eq!(Card::Four, deck[3]);
    }
}
