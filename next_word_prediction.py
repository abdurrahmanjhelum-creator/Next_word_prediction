import random
import json
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, font
from collections import defaultdict
from typing import List, Tuple, Dict, Any

class NextWordPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the NextWordPredictor.
        If model_path is provided, loads an existing model, otherwise creates a new one.
        """
        self.model = {
            'first_words': {},
            'second_words': {},
            'transitions': {}
        }
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, text):
        """
        Train the model on the given text
        """
        # Split text into sentences
        sentences = [s.strip().lower() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 2:
                continue
                
            # Add first word to first_words
            first_word = words[0]
            self.model['first_words'][first_word] = self.model['first_words'].get(first_word, 0) + 1
            
            # Process the rest of the words
            for i in range(1, len(words)):
                prev_word = words[i-1]
                current_word = words[i]
                
                # Add to second words
                if i == 1:
                    if prev_word not in self.model['second_words']:
                        self.model['second_words'][prev_word] = {}
                    self.model['second_words'][prev_word][current_word] = self.model['second_words'][prev_word].get(current_word, 0) + 1
                
                # Add to transitions
                if i > 1:
                    prev_prev_word = words[i-2]
                    key = (prev_prev_word, prev_word)
                    if key not in self.model['transitions']:
                        self.model['transitions'][key] = {}
                    self.model['transitions'][key][current_word] = self.model['transitions'][key].get(current_word, 0) + 1
        
        # Normalize probabilities
        self._normalize_model()
        
    def _normalize_model(self):
        """Convert counts to probabilities"""
        # Normalize first_words
        total = sum(self.model['first_words'].values())
        for word in self.model['first_words']:
            self.model['first_words'][word] /= total
            
        # Normalize second_words
        for word in self.model['second_words']:
            total = sum(self.model['second_words'][word].values())
            for next_word in self.model['second_words'][word]:
                self.model['second_words'][word][next_word] /= total
                
        # Normalize transitions
        for key in self.model['transitions']:
            total = sum(self.model['transitions'][key].values())
            for next_word in self.model['transitions'][key]:
                self.model['transitions'][key][next_word] /= total
    
    def predict_next_word(self, text):
        """
        Predict the next word based on the input text
        Returns a list of (word, probability) tuples sorted by probability
        """
        if not text.strip():
            return []
            
        words = text.lower().split()
        
        # If no input, return most common first words
        if not words:
            if self.model['first_words']:
                return sorted(
                    [(word, prob) for word, prob in self.model['first_words'].items()],
                    key=lambda x: -x[1]
                )[:5]
            return []
            
        # If one word, return likely second words
        if len(words) == 1:
            word = words[0]
            
            # Try exact match first
            if word in self.model['second_words']:
                predictions = sorted(
                    [(w, p) for w, p in self.model['second_words'][word].items()],
                    key=lambda x: -x[1]
                )
                if predictions:
                    return predictions[:5]
            
            # Try partial matches (words that start with the input)
            similar_words = [w for w in self.model['second_words'] if w.startswith(word)]
            if similar_words:
                predictions = []
                for similar in similar_words:
                    predictions.extend([(w, p * 0.7) for w, p in self.model['second_words'][similar].items()])
                if predictions:
                    word_probs = {}
                    for w, p in predictions:
                        word_probs[w] = word_probs.get(w, 0) + p
                    return sorted(word_probs.items(), key=lambda x: -x[1])[:5]
            
            # Try to find words that contain the input
            contains_words = [w for w in self.model['second_words'] if word in w]
            if contains_words:
                predictions = []
                for similar in contains_words:
                    predictions.extend([(w, p * 0.5) for w, p in self.model['second_words'][similar].items()])
                if predictions:
                    word_probs = {}
                    for w, p in predictions:
                        word_probs[w] = word_probs.get(w, 0) + p
                    return sorted(word_probs.items(), key=lambda x: -x[1])[:5]
        
        # If multiple words, use the last two for prediction
        if len(words) >= 2:
            prev_prev_word = words[-2]
            prev_word = words[-1]
            key = (prev_prev_word, prev_word)
            
            # Try exact match for last two words
            if key in self.model['transitions']:
                predictions = sorted(
                    [(w, p) for w, p in self.model['transitions'][key].items()],
                    key=lambda x: -x[1]
                )
                if predictions:
                    return predictions[:5]
            
            # Try with just the last word
            if prev_word in self.model['second_words']:
                predictions = sorted(
                    [(w, p * 0.7) for w, p in self.model['second_words'][prev_word].items()],
                    key=lambda x: -x[1]
                )
                if predictions:
                    return predictions[:5]
            
            # Try with partial match for the last word
            similar_last_words = [w for w in self.model['second_words'] if w.startswith(prev_word)]
            if similar_last_words:
                predictions = []
                for similar in similar_last_words:
                    predictions.extend([(w, p * 0.5) for w, p in self.model['second_words'][similar].items()])
                if predictions:
                    word_probs = {}
                    for w, p in predictions:
                        word_probs[w] = word_probs.get(w, 0) + p
                    return sorted(word_probs.items(), key=lambda x: -x[1])[:5]
        
        # If still no predictions, try to find any word that follows the last word
        last_word = words[-1]
        if last_word in self.model['second_words']:
            predictions = sorted(
                [(w, p * 0.3) for w, p in self.model['second_words'][last_word].items()],
                key=lambda x: -x[1]
            )
            if predictions:
                return predictions[:3]  # Return fewer options for less confident predictions
        
        # As a last resort, return most common words
        if self.model['first_words']:
            return sorted(
                [(w, p * 0.2) for w, p in self.model['first_words'].items()],
                key=lambda x: -x[1]
            )[:3]
            
        return []
    
    def save_model(self, filepath):
        """Save the model to a JSON file"""
        # Convert tuples to strings for JSON serialization
        serializable_model = {
            'first_words': self.model['first_words'],
            'second_words': self.model['second_words'],
            'transitions': {f"{k[0]},{k[1]}": v for k, v in self.model['transitions'].items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_model, f, indent=2)
    
    def load_model(self, filepath):
        """Load a model from a JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_model = json.load(f)
        
        # Convert string keys back to tuples
        self.model = {
            'first_words': loaded_model['first_words'],
            'second_words': loaded_model['second_words'],
            'transitions': {tuple(k.split(',')): v for k, v in loaded_model['transitions'].items()}
        }

def get_extended_training_data():
    """Return a more comprehensive training dataset"""
    return """
    My name is Ali and he is. its colour is beautiful . there is no reality.  I am a student at Delhi Technological University.
    I am studying computer science and engineering.
    My friend Ali is very good at programming.
    Ali and his friends are working on a project.
    I met Ali at the university yesterday.
    hello this is hamza.
    Ali's favorite subject is artificial intelligence.Ghulam mustafa is good teacher.Mustafa khizer is CR. where is your brother.then he is going.
    
    
    I am learning about machine learning and artificial intelligence.
    Artificial intelligence is changing the world in many ways.
    Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
    Deep learning is a type of machine learning that uses neural networks with multiple layers.
    I love programming in Python because it's easy to learn and powerful.
    Python is a great language for data science and machine learning.
    Data science involves statistics, programming, and domain expertise.
    I enjoy learning new technologies and improving my skills.
    
    The quick brown fox jumps over the lazy dog.
    She sells seashells by the seashore.
    How much wood would a woodchuck chuck if a woodchuck could chuck wood?
    Peter Piper picked a peck of pickled peppers.
    The cat in the hat came back the very next day.
    To be or not to be, that is the question.
    All that glitters is not gold.
    A picture is worth a thousand words.
    Actions speak louder than words.
    The early bird catches the worm.
    what is your name.
    Don't count your chickens before they hatch.
    Every cloud has a silver lining.
    Fortune favors the bold.
    Great minds think alike.
    then where will you go now
    Honesty is the best policy.
    If it ain't broke, don't fix it.
    Knowledge is power.
    Look before you leap.
    Money doesn't grow on trees.
    No pain, no gain.
    Practice makes perfect.
    Rome wasn't built in a day.
    The pen is mightier than the sword.
    When in Rome, do as the Romans do.
    You can't judge a book by its cover.
    A journey of a thousand miles begins with a single step.
    Better late than never.
    Don't put all your eggs in one basket.
    Every dog has its day.
    Good things come to those who wait.
    If you can't beat them, join them.
    It's no use crying over spilt milk.
    Let sleeping dogs lie.
    Necessity is the mother of invention.
    Out of sight, out of mind.
    The grass is always greener on the other side.
    Two wrongs don't make a right.
    When the going gets tough, the tough get going.
    You can lead a horse to water, but you can't make it drink.
    A bird in the hand is worth two in the bush.
    Don't bite the hand that feeds you.
    Every man has his price.
    Haste makes waste.
    If the shoe fits, wear it.
    It's better to be safe than sorry.
    Like father, like son.
    No news is good news.
    The early bird gets the worm.
    There's no place like home.
    When the cat's away, the mice will play.
    You can't have your cake and eat it too.
    A friend in need is a friend indeed.
    Don't judge a book by its cover.
    Every cloud has a silver lining.
    He who laughs last laughs longest.
    If you want something done right, do it yourself.
    It's never too late to mend.
    Look before you leap.
    One man's trash is another man's treasure.
    The best things in life are free.
    There's no time like the present.
    Where there's smoke, there's fire.
    You can't make an omelette without breaking eggs.
    A penny saved is a penny earned.
    Don't put off until tomorrow what you can do today.
    Home is where the heart is.
    If you can't stand the heat, get out of the kitchen.
    It's always darkest before the dawn.
    Make hay while the sun shines.
    Out of sight, out of mind.
    The best of both worlds.
    There's no such thing as a free lunch.
    Where there's a will, there's a way.
    You can't please everyone.
    A picture is worth a thousand words.
    Don't throw the baby out with the bathwater.
    Every rose has its thorn.
    Hope for the best, prepare for the worst.
    If you play with fire, you'll get burned.
    It's better to give than to receive.
    Many hands make light work.
    Practice what you preach.
    The early bird catches the worm.
    There's no place like home.
    You can't teach an old dog new tricks.
    A rolling stone gathers no moss.
    Easy come, easy go.
    First things first.
    Ignorance is bliss.
    It's a piece of cake.
    Money doesn't grow on trees.
    Put your best foot forward.
    The grass is always greener on the other side.
    There's no time like the present.
    You can't win them all.
    A stitch in time saves nine.
    Every cloud has a silver lining.
    Fortune favors the bold.
    It takes two to tango.
    Let bygones be bygones.
    No pain, no gain.
    Rome wasn't built in a day.
    The pen is mightier than the sword.
    Those who live in glass houses shouldn't throw stones.
    You reap what you sow.
    """

def train_on_sample_text():
    """Train the model on some sample text"""
    predictor = NextWordPredictor()
    
    # Train on the extended dataset
    sample_text = get_extended_training_data()
    predictor.train(sample_text)
    
    # Also add some common names and words
    names = "Ali is a common name. Muhammad is also very common. Ahmed and Sara are popular names too. " \
            "John, Michael, David, Sarah, and Emily are common English names. " \
            "Ali went to the park. Muhammad likes to read. Sara is a good student. " \
            "Ahmed plays football. John works at a company. Emily loves music."
    predictor.train(names)
    
    # Save the trained model
    predictor.save_model('word_prediction_model.json')
    return predictor

class NextWordPredictorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Next Word Predictor")
        self.root.geometry("900x600")
        self.root.minsize(800, 500)
        
        # Configure grid weights for responsiveness
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Set theme colors
        self.bg_color = "#f0f0f0"
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#166088"
        self.accent_color = "#17a2b8"
        self.text_color = "#333333"
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize predictor
        self.predictor = train_on_sample_text()
        
        # Configure styles
        self.setup_styles()
        
        # Create main container with grid layout
        self.create_widgets()
        
        # Bind keyboard shortcuts
        self.setup_bindings()
        
        # Set focus to input field
        self.input_entry.focus()
        
        # Store prediction buttons for easy clearing
        self.prediction_buttons = []
        
    def setup_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        
        # Configure colors
        style.theme_use('clam')
        
        # Configure main frame style
        style.configure('Main.TFrame', background=self.bg_color)
        
        # Configure card frame style
        style.configure('Card.TFrame', background='white', relief='solid', borderwidth=1)
        
        # Configure label styles
        style.configure('Title.TLabel', 
                       background=self.bg_color,
                       foreground=self.secondary_color,
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=self.bg_color,
                       foreground=self.text_color,
                       font=('Segoe UI', 11))
        
        style.configure('Prediction.TLabel',
                       background='white',
                       foreground=self.text_color,
                       font=('Segoe UI', 10))
        
        # Configure button styles
        style.configure('Primary.TButton',
                       background=self.primary_color,
                       foreground='white',
                       borderwidth=1,
                       focusthickness=3,
                       focuscolor='none')
        
        style.configure('Prediction.TButton',
                       background='white',
                       foreground=self.primary_color,
                       borderwidth=1,
                       relief='solid',
                       font=('Segoe UI', 10, 'bold'),
                       padding=5)
        
        style.map('Prediction.TButton',
                 background=[('active', '#e6f2ff')],
                 foreground=[('active', self.secondary_color)])
        
        # Configure entry style
        style.configure('Input.TEntry',
                       fieldbackground='white',
                       borderwidth=2,
                       relief='solid',
                       font=('Segoe UI', 12))
        
        # Configure status bar style
        style.configure('Status.TLabel',
                       background='white',
                       foreground=self.text_color,
                       font=('Segoe UI', 9),
                       padding=5)
        
    def create_widgets(self):
        """Create and arrange all widgets"""
        # Main container
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Configure grid weights for main container
        main_container.grid_columnconfigure(0, weight=1)
        for i in range(4):
            main_container.grid_rowconfigure(i, weight=1 if i == 3 else 0)
        
        # Header section
        self.create_header(main_container)
        
        # Input section
        self.create_input_section(main_container)
        
        # Predictions section
        self.create_predictions_section(main_container)
        
        # Status bar
        self.create_status_bar()
        
    def create_header(self, parent):
        """Create the header/title section"""
        header_frame = ttk.Frame(parent, style='Main.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                               text="Next Word Predictor", 
                               style='Title.TLabel')
        title_label.pack(anchor="w")
        
        subtitle_label = ttk.Label(header_frame,
                                  text="Type your text and get AI-powered word suggestions",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(anchor="w", pady=(5, 0))
        
    def create_input_section(self, parent):
        """Create the input text section"""
        input_card = ttk.Frame(parent, style='Card.TFrame')
        input_card.grid(row=1, column=0, sticky="ew", pady=(0, 15), padx=5)
        input_card.grid_columnconfigure(0, weight=1)
        
        # Input label
        input_label = ttk.Label(input_card,
                               text="Start typing your text:",
                               style='Subtitle.TLabel',
                               background='white')
        input_label.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))
        
        # Input entry
        input_container = ttk.Frame(input_card, style='Card.TFrame')
        input_container.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        input_container.grid_columnconfigure(0, weight=1)
        
        self.input_var = tk.StringVar()
        self.input_var.trace('w', self.on_text_change)
        self.input_entry = ttk.Entry(input_container,
                                    textvariable=self.input_var,
                                    style='Input.TEntry',
                                    font=('Segoe UI', 12))
        self.input_entry.grid(row=0, column=0, sticky="ew", ipady=10)
        
        # Instructions
        instruction_label = ttk.Label(input_card,
                                     text="Press Enter to accept the top prediction, or click any suggestion button",
                                     style='Prediction.TLabel',
                                     background='white',
                                     font=('Segoe UI', 9))
        instruction_label.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 15))
        
    def create_predictions_section(self, parent):
        """Create the predictions display section"""
        self.predictions_card = ttk.Frame(parent, style='Card.TFrame')
        self.predictions_card.grid(row=2, column=0, sticky="nsew", pady=(0, 15), padx=5)
        self.predictions_card.grid_columnconfigure(0, weight=1)
        self.predictions_card.grid_rowconfigure(1, weight=1)
        
        # Predictions title
        predictions_label = ttk.Label(self.predictions_card,
                                     text="Word Suggestions:",
                                     style='Subtitle.TLabel',
                                     background='white')
        predictions_label.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))
        
        # Predictions container
        self.predictions_container = ttk.Frame(self.predictions_card, style='Card.TFrame')
        self.predictions_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # Configure grid for prediction buttons
        for i in range(5):
            self.predictions_container.grid_columnconfigure(i, weight=1)
        
    def create_status_bar(self):
        """Create the status bar at the bottom"""
        self.status_bar = ttk.Frame(self.root, relief='sunken', borderwidth=1)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Model trained with extended dataset")
        
        status_label = ttk.Label(self.status_bar,
                                textvariable=self.status_var,
                                style='Status.TLabel',
                                anchor='w')
        status_label.pack(side=tk.LEFT, fill='x', expand=True)
        
        # Add word count label
        self.word_count_var = tk.StringVar()
        self.word_count_var.set("Words: 0")
        word_count_label = ttk.Label(self.status_bar,
                                    textvariable=self.word_count_var,
                                    style='Status.TLabel',
                                    anchor='e')
        word_count_label.pack(side=tk.RIGHT, padx=10)
        
    def setup_bindings(self):
        """Setup keyboard bindings"""
        self.root.bind('<Return>', self.add_prediction)
        self.root.bind('<Escape>', lambda e: self.clear_input())
        self.input_entry.bind('<KeyRelease>', self.update_word_count)
        
    def on_text_change(self, *args):
        """Handle text changes in the input field"""
        text = self.input_var.get()
        
        # Update word count
        self.update_word_count()
        
        if not text.strip():
            self.clear_predictions()
            self.update_status("Ready - Type to get predictions")
            return
            
        try:
            # Get predictions
            predictions = self.predictor.predict_next_word(text)
            
            # Clear previous predictions
            self.clear_predictions()
            
            # Update status
            if predictions:
                self.update_status(f"Found {len(predictions)} prediction(s)")
            else:
                self.update_status("No predictions available")
            
            # Display predictions
            if predictions:
                self.display_predictions(predictions)
            else:
                self.show_no_predictions()
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.clear_predictions()
            
    def display_predictions(self, predictions):
        """Display prediction buttons"""
        self.prediction_buttons = []
        
        for i, (word, prob) in enumerate(predictions[:5], 1):
            # Create a styled button for each prediction
            btn_frame = ttk.Frame(self.predictions_container, style='Card.TFrame')
            btn_frame.grid(row=0, column=i-1, padx=5, pady=5, sticky="nsew")
            btn_frame.grid_columnconfigure(0, weight=1)
            
            # Button text with probability
            btn_text = f"{word}"
            prob_text = f"{prob*100:.1f}%"
            
            # Create button
            btn = ttk.Button(btn_frame,
                            text=btn_text,
                            style='Prediction.TButton',
                            command=lambda w=word: self.select_prediction(w))
            btn.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
            
            # Add probability label
            prob_label = ttk.Label(btn_frame,
                                  text=prob_text,
                                  style='Prediction.TLabel',
                                  foreground=self.accent_color,
                                  font=('Segoe UI', 9))
            prob_label.grid(row=1, column=0, pady=(0, 10))
            
            self.prediction_buttons.append((btn_frame, btn, prob_label))
            
            # Add keyboard shortcuts (1-5)
            if i <= 5:
                self.root.bind(str(i), lambda e, w=word: self.select_prediction(w))
            
    def select_prediction(self, word):
        """Handle prediction selection"""
        current_text = self.input_var.get()
        
        # If the last character is a space, add as new word
        if current_text.endswith(' '):
            new_text = f"{current_text}{word} "
        else:
            # Replace last word with prediction
            words = current_text.split()
            if words:
                words[-1] = word
                new_text = ' '.join(words) + ' '
            else:
                new_text = f"{word} "
        
        self.input_var.set(new_text)
        self.input_entry.icursor(tk.END)
        self.input_entry.focus()
        
        # Clear keyboard shortcuts
        for i in range(1, 6):
            self.root.unbind(str(i))
        
        self.on_text_change()  # Update predictions
        
    def add_prediction(self, event=None):
        """Handle Enter key press to add first prediction"""
        text = self.input_var.get().strip()
        if not text:
            return
            
        predictions = self.predictor.predict_next_word(text)
        if predictions:
            self.select_prediction(predictions[0][0])
            return "break"  # Prevent default Enter behavior
        
    def clear_predictions(self):
        """Clear prediction buttons"""
        # Clear keyboard shortcuts
        for i in range(1, 6):
            self.root.unbind(str(i))
            
        # Destroy all prediction widgets
        for widget in self.predictions_container.winfo_children():
            widget.destroy()
            
        self.prediction_buttons.clear()
        
    def show_no_predictions(self):
        """Display message when no predictions are available"""
        no_pred_label = ttk.Label(self.predictions_container,
                                 text="No predictions available. Try typing more text.",
                                 style='Prediction.TLabel',
                                 foreground="#666666")
        no_pred_label.grid(row=0, column=0, columnspan=5, padx=20, pady=20)
        
    def clear_input(self):
        """Clear the input field"""
        self.input_var.set("")
        self.input_entry.focus()
        
    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
        
    def update_word_count(self, event=None):
        """Update the word count display"""
        text = self.input_var.get()
        words = text.split()
        self.word_count_var.set(f"Words: {len(words)}")

def interactive_demo():
    """Start the Tkinter UI"""
    root = tk.Tk()
    app = NextWordPredictorUI(root)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    print("Starting Next Word Predictor...")
    print("Please wait while the model is being trained...")
    interactive_demo()