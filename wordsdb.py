import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any


class WordsDatabase:
    """
    Database to track the performance of words when added to a prefix.
    Stores word statistics and allows querying for top-performing words.
    """

    def __init__(self, db_path: str = "word_performance.db"):
        """Initialize the database, creating tables if they don't exist."""
        self.db_path = db_path
        self.conn = None
        self.initialize_db()

    def initialize_db(self):
        """Create the database tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()

            # Create table for word performance
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS word_performance (
                id INTEGER PRIMARY KEY,
                word TEXT NOT NULL,
                position TEXT NOT NULL,
                benign_score REAL NOT NULL,
                improvement REAL NOT NULL,
                token_count INTEGER NOT NULL,
                combined_score REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create table for word statistics (aggregated data)
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS word_stats (
                word TEXT PRIMARY KEY,
                avg_improvement REAL NOT NULL,
                max_improvement REAL NOT NULL,
                avg_token_count REAL NOT NULL,
                min_token_count INTEGER NOT NULL,
                use_count INTEGER NOT NULL,
                best_position TEXT NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            self.conn.commit()
            print(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def record_word_performance(
        self,
        word: str,
        position: str,
        benign_score: float,
        improvement: float,
        token_count: int,
        combined_score: float,
    ):
        """Record the performance of a word when added to a prefix."""
        if self.conn is None:
            self.initialize_db()

        try:
            cursor = self.conn.cursor()

            # Insert performance record
            cursor.execute(
                """
            INSERT INTO word_performance 
            (word, position, benign_score, improvement, token_count, combined_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    word,
                    position,
                    benign_score,
                    improvement,
                    token_count,
                    combined_score,
                ),
            )

            # Update statistics
            cursor.execute(
                """
            INSERT INTO word_stats 
            (word, avg_improvement, max_improvement, avg_token_count, min_token_count, use_count, best_position)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(word) DO UPDATE SET
                avg_improvement = (avg_improvement * use_count + ?) / (use_count + 1),
                max_improvement = MAX(max_improvement, ?),
                avg_token_count = (avg_token_count * use_count + ?) / (use_count + 1),
                min_token_count = MIN(min_token_count, ?),
                use_count = use_count + 1,
                best_position = CASE WHEN ? > max_improvement THEN ? ELSE best_position END,
                last_updated = CURRENT_TIMESTAMP
            """,
                (
                    word,
                    improvement,
                    improvement,
                    token_count,
                    token_count,
                    position,
                    improvement,
                    improvement,
                    token_count,
                    token_count,
                    improvement,
                    position,
                ),
            )

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error recording word performance: {e}")
            # Still try to continue without failing

    def record_gcg_token_performance(
        self, token: str, improvement: float, benign_score: float
    ):
        """Record the performance of a token from GCG attack."""
        if self.conn is None:
            self.initialize_db()

        try:
            cursor = self.conn.cursor()

            position = "gcg"
            token_count = 1
            combined_score = improvement  # Use improvement as a proxy for combined_score

            # Insert performance record
            cursor.execute(
                """
            INSERT INTO word_performance
            (word, position, benign_score, improvement, token_count, combined_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    token,
                    position,
                    benign_score,
                    improvement,
                    token_count,
                    combined_score,
                ),
            )

            # Update statistics
            cursor.execute(
                """
            INSERT INTO word_stats
            (word, avg_improvement, max_improvement, avg_token_count, min_token_count, use_count, best_position)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(word) DO UPDATE SET
                avg_improvement = (avg_improvement * use_count + ?) / (use_count + 1),
                max_improvement = MAX(max_improvement, ?),
                avg_token_count = (avg_token_count * use_count + ?) / (use_count + 1),
                min_token_count = MIN(min_token_count, ?),
                use_count = use_count + 1,
                best_position = CASE WHEN ? > max_improvement THEN ? ELSE best_position END,
                last_updated = CURRENT_TIMESTAMP
            """,
                (
                    token,
                    improvement,
                    improvement,
                    token_count,
                    token_count,
                    position,
                    improvement,
                    improvement,
                    token_count,
                    token_count,
                    improvement,
                    position,
                ),
            )

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error recording GCG token performance: {e}")

    def get_top_words(
        self,
        limit: int = 20,
        min_uses: int = 2,
        sort_by: str = "improvement",
        token_weight: float = 0.0,
    ) -> List[str]:
        """
        Get the top-performing words based on selected criteria.

        Parameters:
        -----------
        limit: Maximum number of words to return
        min_uses: Minimum number of uses a word must have to be considered
        sort_by: How to sort the results - options: "improvement", "tokens", "combined"
        token_weight: When sort_by="combined", weight for token count vs improvement (0-1)

        Returns:
        --------
        List of words matching the criteria
        """
        if self.conn is None:
            self.initialize_db()

        try:
            cursor = self.conn.cursor()

            # Different sorting strategies
            if sort_by == "tokens":
                # Sort by token count (ascending) then by improvement (descending)
                cursor.execute(
                    """
                SELECT word FROM word_stats 
                WHERE use_count >= ? AND avg_improvement > 0
                ORDER BY min_token_count ASC, avg_improvement DESC
                LIMIT ?
                """,
                    (min_uses, limit),
                )
            elif sort_by == "combined":
                # Get all qualifying words with their stats
                cursor.execute(
                    """
                SELECT word, avg_improvement, min_token_count 
                FROM word_stats 
                WHERE use_count >= ? AND avg_improvement > 0
                """,
                    (min_uses,),
                )

                # Calculate combined scores
                results = cursor.fetchall()
                if not results:
                    return []

                # Normalize values
                max_improvement = max(row[1] for row in results)
                max_tokens = max(row[2] for row in results)

                # Calculate combined score for each word
                scored_words = []
                for row in results:
                    word = row[0]
                    norm_improvement = row[1] / max_improvement if max_improvement > 0 else 0
                    norm_tokens = 1 - (
                        row[2] / max_tokens if max_tokens > 0 else 0
                    )  # Invert so lower is better
                    combined_score = (
                        1 - token_weight
                    ) * norm_improvement + token_weight * norm_tokens
                    scored_words.append((word, combined_score))

                # Sort by combined score and return top words
                scored_words.sort(key=lambda x: x[1], reverse=True)
                return [word for word, _ in scored_words[:limit]]
            else:
                # Default: sort by improvement
                cursor.execute(
                    """
                SELECT word FROM word_stats 
                WHERE use_count >= ? AND avg_improvement > 0
                ORDER BY avg_improvement DESC
                LIMIT ?
                """,
                    (min_uses, limit),
                )

            results = cursor.fetchall()
            return [row[0] for row in results]
        except sqlite3.Error as e:
            print(f"Error getting top words: {e}")
            return []

    def get_word_stats(self, word: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific word."""
        if self.conn is None:
            self.initialize_db()

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
            SELECT word, avg_improvement, max_improvement, avg_token_count, min_token_count, use_count, best_position
            FROM word_stats 
            WHERE word = ?
            """,
                (word,),
            )

            result = cursor.fetchone()
            if result:
                return {
                    "word": result[0],
                    "avg_improvement": result[1],
                    "max_improvement": result[2],
                    "avg_token_count": result[3],
                    "min_token_count": result[4],
                    "use_count": result[5],
                    "best_position": result[6],
                }
            return None
        except sqlite3.Error as e:
            print(f"Error getting word stats: {e}")
            return None

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
