"""
Autocomplete System using SQLite3 for Persistence with N-gram optimization

This module implements an autocomplete system that learns word sequences from training sentences
and predicts the most likely next word based on the learned patterns. It uses SQLite3 for
persistent storage of word mappings and predictions.
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Union

class AutoComplete:
    """
    An autocomplete system that trains on text data and predicts subsequent words using N-gram model.
    
    The system works by:
    1. Building N-gram maps that track how often each N-gram is followed by another word
    2. Maintaining predictions for the most likely next word for each N-gram
    3. Storing all data in an SQLite database for persistence
    """

    def __init__(self, n=2) -> None:
        """
        Initialize the AutoComplete system and set up the database.
        
        Creates an SQLite database connection and initializes required tables
        (NGramMap and NGramPrediction) if they don't already exist. These tables
        store the N-gram transition mappings and precomputed predictions respectively.
        """
        self.n = n
        # Establish database connection with autocommit enabled
        self.conn: sqlite3.Connection = sqlite3.connect("autocompleteDB.sqlite3", autocommit=True)
        cursor: sqlite3.Cursor = self.conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE name='NGramMap'")
        tables_exist: Optional[Tuple[str]] = cursor.fetchone()
        
        if not tables_exist:
            # Create tables if they don't exist
            cursor.execute("CREATE TABLE NGramMap(name TEXT, value TEXT)")
            cursor.execute("CREATE TABLE NGramPrediction(name TEXT, value TEXT)")
            
            # Initialize with empty dictionaries
            cursor.execute("INSERT INTO NGramMap VALUES (?, ?)", ("ngramsmap", "{}"))
            cursor.execute("INSERT INTO NGramPrediction VALUES (?, ?)", ("ngrampredictions", "{}"))

    def generate_ngrams(self, words_list: List[str]) -> List[Tuple[str]]:
        """
        Generate N-grams from a list of words.
        """
        ngrams = []
        for i in range(len(words_list) - self.n + 1):
            ngrams.append(tuple(words_list[i:i+self.n]))
        return ngrams

    def train(self, sentence: str) -> str:
        """
        Train the autocomplete system with a single sentence.
        
        Processes the input sentence to update:
        1. N-gram transition counts (NGramMap)
        2. Most likely next word predictions (NGramPrediction)
        
        Args:
            sentence: A string containing the training text. Words should be space-separated.
            
        Returns:
            Confirmation message indicating training completion.
        """
        cursor: sqlite3.Cursor = self.conn.cursor()
        
        # Split sentence into individual words
        words_list: List[str] = sentence.split(" ")
        
        # Retrieve existing N-gram map and predictions from database
        cursor.execute("SELECT value FROM NGramMap WHERE name='ngramsmap'")
        ngrams_map_str: str = cursor.fetchone()[0]
        ngrams_map: Dict[Tuple[str], Dict[str, int]] = json.loads(ngrams_map_str, object_hook=lambda d: {tuple(k.split()): v for k, v in d.items()})
        
        cursor.execute("SELECT value FROM NGramPrediction WHERE name='ngrampredictions'")
        predictions_str: str = cursor.fetchone()[0]
        predictions: Dict[Tuple[str], Dict[str, Union[str, int]]] = json.loads(predictions_str, object_hook=lambda d: {tuple(k.split()): v for k, v in d.items()})
        
        # Generate N-grams
        ngrams = self.generate_ngrams(words_list)
        
        # Process each N-gram and the next word
        for i in range(len(ngrams) - 1):
            curr_ngram: Tuple[str] = ngrams[i]
            next_word: str = words_list[i + self.n]
            
            # Update N-gram transition counts
            if curr_ngram not in ngrams_map:
                ngrams_map[curr_ngram] = {}
            
            if next_word not in ngrams_map[curr_ngram]:
                ngrams_map[curr_ngram][next_word] = 1
            else:
                ngrams_map[curr_ngram][next_word] += 1
            
            # Update predictions with most frequent next word
            if curr_ngram not in predictions:
                predictions[curr_ngram] = {
                    'completion_word': next_word,
                    'completion_count': 1
                }
            else:
                # Update if current next word is more frequent
                if ngrams_map[curr_ngram][next_word] > predictions[curr_ngram]['completion_count']:
                    predictions[curr_ngram]['completion_word'] = next_word
                    predictions[curr_ngram]['completion_count'] = ngrams_map[curr_ngram][next_word]
        
        # Save updated data back to database
        updated_ngrams_map: str = json.dumps({ ' '.join(k): v for k, v in ngrams_map.items() })
        updated_predictions: str = json.dumps({ ' '.join(k): v for k, v in predictions.items() })
        
        cursor.execute("UPDATE NGramMap SET value = ? WHERE name='ngramsmap'", (updated_ngrams_map,))
        cursor.execute("UPDATE NGramPrediction SET value = ? WHERE name='ngrampredictions'", (updated_predictions,))
        
        return "training complete"

    def predict(self, words: str) -> Optional[str]:
        """
        Predict the most likely next word for a given input sequence of words.
        
        Args:
            words: The input sequence of words to generate a completion for.
            
        Returns:
            The most likely next word, or None if no prediction exists.
            
        Raises:
            KeyError: If the input sequence of words has no entries in the prediction database.
        """
        cursor: sqlite3.Cursor = self.conn.cursor()
        
        # Retrieve predictions from database
        cursor.execute("SELECT value FROM NGramPrediction WHERE name='ngrampredictions'")
        predictions_str: str = cursor.fetchone()[0]
        predictions: Dict[Tuple[str], Dict[str, Union[str, int]]] = json.loads(predictions_str, object_hook=lambda d: {tuple(k.split()): v for k, v in d.items()})
        
        input_words = words.lower().split()
        for i in range(len(input_words), max(0, len(input_words) - self.n + 1), -1):
            curr_ngram = tuple(input_words[i - self.n:i])
            if curr_ngram in predictions:
                return str(predictions[curr_ngram]['completion_word'])
        return None


if __name__ == "__main__":
    # Example usage
    training_sentence: str = (
        "It is not enough to just know how tools work and what they worth, "
        "we have got to learn how to use them and to use them well. And with "
        "all these new weapons in your arsenal, we would better get those profits fired up"
    )
    
    # Initialize and train the autocomplete system
    autocomplete: AutoComplete = AutoComplete(n=2)
    autocomplete.train(training_sentence)
    
    # Test prediction
    test_words: str = "to use"
    prediction: Optional[str] = autocomplete.predict(test_words)
    print(f"Prediction for '{test_words}': {prediction}")