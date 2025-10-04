import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import deque

class ConversationMemory:
    """Manages persistent memory and conversation history"""
    
    def __init__(self, db_path: str, max_history: int = 20):
        self.db_path = db_path
        self.max_history = max_history
        self.short_term_memory = deque(maxlen=10)
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                assistant_response TEXT,
                emotion TEXT,
                topics TEXT
            )
        ''')
        
        # User information and preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_info (
                key TEXT PRIMARY KEY,
                value TEXT,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL DEFAULT 1.0
            )
        ''')
        
        # Long-term memories
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                importance REAL DEFAULT 0.5,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Reminders and tasks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                due_date DATETIME,
                completed BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_conversation(self, user_input: str, response: str, emotion: str = "neutral", topics: List[str] = None):
        """Save a conversation with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        topics_str = json.dumps(topics) if topics else "[]"
        cursor.execute(
            'INSERT INTO conversations (user_input, assistant_response, emotion, topics) VALUES (?, ?, ?, ?)',
            (user_input, response, emotion, topics_str)
        )
        
        self.short_term_memory.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.datetime.now()
        })
        
        conn.commit()
        conn.close()
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, assistant_response, timestamp, emotion, topics 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "user": row[0],
            "assistant": row[1], 
            "timestamp": row[2],
            "emotion": row[3],
            "topics": json.loads(row[4]) if row[4] else []
        } for row in reversed(results)]
    
    def save_user_info(self, key: str, value: str, category: str = "general", confidence: float = 1.0):
        """Save information about the user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT OR REPLACE INTO user_info (key, value, category, confidence) VALUES (?, ?, ?, ?)',
            (key, value, category, confidence)
        )
        
        conn.commit()
        conn.close()
    
    def get_user_info(self, key: str = None, category: str = None) -> Dict[str, Any]:
        """Retrieve user information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if key:
            cursor.execute('SELECT value, confidence FROM user_info WHERE key = ?', (key,))
            result = cursor.fetchone()
            conn.close()
            return {"value": result[0], "confidence": result[1]} if result else None
        
        if category:
            cursor.execute('SELECT key, value, confidence FROM user_info WHERE category = ?', (category,))
        else:
            cursor.execute('SELECT key, value, confidence, category FROM user_info')
        
        results = cursor.fetchall()
        conn.close()
        
        return {row[0]: {"value": row[1], "confidence": row[2]} for row in results}
    
    def add_memory(self, content: str, importance: float = 0.5, category: str = "general"):
        """Add a long-term memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO memories (content, importance, category) VALUES (?, ?, ?)',
            (content, importance, category)
        )
        
        conn.commit()
        conn.close()
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories by content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT content, importance, category, timestamp 
            FROM memories 
            WHERE content LIKE ? 
            ORDER BY importance DESC, timestamp DESC 
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "content": row[0],
            "importance": row[1],
            "category": row[2],
            "timestamp": row[3]
        } for row in results]