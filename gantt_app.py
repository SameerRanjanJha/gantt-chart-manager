#!/usr/bin/env python3
"""
Gantt Chart Desktop Application
A standalone desktop app for creating and managing Gantt charts from CSV files.

Dependencies:
- pip install PySimpleGUI matplotlib pandas

To create executable:
- pyinstaller --onefile gantt_app.py

Author: Claude AI Assistant
License: MIT
Version: 1.0.0
"""

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import csv
import datetime
import os
import logging
import sys
from typing import Optional, Tuple, List, Dict, Any
import json
import copy

# Configure logging
logging.basicConfig(
    filename='gantt_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class UndoRedoManager:
    """Manages undo/redo operations for the application."""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.history: List[pd.DataFrame] = []
        self.current_index = -1
        
    def save_state(self, df: pd.DataFrame):
        """Save current state for undo/redo."""
        if df is None or df.empty:
            return
            
        # Remove any states after current index
        self.history = self.history[:self.current_index + 1]
        
        # Add new state
        self.history.append(df.copy())
        self.current_index += 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1
            
    def can_undo(self) -> bool:
        return self.current_index > 0
        
    def can_redo(self) -> bool:
        return self.current_index < len(self.history) - 1
        
    def undo(self) -> Optional[pd.DataFrame]:
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index].copy()
        return None
        
    def redo(self) -> Optional[pd.DataFrame]:
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index].copy()
        return None

class GanttChart:
    """Handles the Gantt chart visualization and interactions."""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.canvas = None
        self.colors = plt.cm.Set3(range(12))  # Color palette
        self.owner_colors = {}
        self.dragging = False
        self.drag_task = None
        self.drag_handle = None  # 'start' or 'end'
        self.selected_task = None
        
    def create_figure(self):
        """Create the matplotlib figure."""
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.patch.set_facecolor('white')
        return self.fig
        
    def assign_colors(self, owners: List[str]):
        """Assign colors to owners."""
        unique_owners = list(set(owners))
        for i, owner in enumerate(unique_owners):
            self.owner_colors[owner] = self.colors[i % len(self.colors)]
            
    def plot_gantt(self, df: pd.DataFrame):
        """Plot the Gantt chart."""
        if df is None or df.empty:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No tasks to display\nAdd tasks via CSV or Add Task button', 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.fig.canvas.draw()
            return
            
        self.ax.clear()
        
        # Assign colors to owners
        self.assign_colors(df['owner'].tolist())
        
        # Calculate date range with better padding
        min_date = df['start'].min() - pd.Timedelta(days=3)
        max_date = df['end'].max() + pd.Timedelta(days=3)
        
        # Plot bars
        y_pos = range(len(df))
        for i, (idx, row) in enumerate(df.iterrows()):
            start_date = row['start']
            end_date = row['end']
            duration = (end_date - start_date).days
            
            color = self.owner_colors[row['owner']]
            bar = self.ax.barh(i, duration, left=start_date, height=0.6, 
                             color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add task name inside bar if it fits, otherwise place it to the left
            if duration > 7:  # Only if bar is wide enough
                self.ax.text(start_date + pd.Timedelta(days=duration/2), i, 
                           row['name'], ha='center', va='center', fontweight='bold', fontsize=9)
            else:
                # Place text to the left of the bar if it's too narrow
                self.ax.text(start_date - pd.Timedelta(days=1), i, 
                           row['name'], ha='right', va='center', fontweight='bold', fontsize=9)
        
        # Configure axes
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(df['name'].tolist(), fontsize=10)
        self.ax.set_xlabel('Timeline', fontsize=12)
        self.ax.set_ylabel('Tasks', fontsize=12)
        self.ax.set_title('Gantt Chart', fontsize=14, fontweight='bold', pad=20)
        
        # Format x-axis
        self.ax.set_xlim(min_date, max_date)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Create legend with better positioning
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=owner) 
                          for owner, color in self.owner_colors.items()]
        
        # Position legend outside the plot area
        legend = self.ax.legend(handles=legend_elements, 
                               loc='center left', 
                               bbox_to_anchor=(1.02, 0.5),
                               frameon=True,
                               fancybox=True,
                               shadow=True,
                               fontsize=10)
        
        # Set legend frame properties
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('gray')
        
        # Grid
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout to prevent overlap
        # Use subplots_adjust instead of tight_layout for better control
        self.fig.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.2)
        
        self.fig.canvas.draw()

class CSVValidator:
    """Handles CSV validation and parsing."""
    
    REQUIRED_COLUMNS = ['name', 'start', 'end', 'owner']
    
    @staticmethod
    def validate_csv_structure(filepath: str) -> Tuple[bool, str]:
        """Validate CSV has required columns."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
            if header is None:
                return False, "CSV file is empty"
                
            # Normalize column names (lowercase, strip whitespace)
            header_normalized = [col.lower().strip() for col in header]
            required_normalized = [col.lower() for col in CSVValidator.REQUIRED_COLUMNS]
            
            missing_cols = [col for col in required_normalized if col not in header_normalized]
            if missing_cols:
                return False, f"Missing required columns: {', '.join(missing_cols)}"
                
            # Check for extra columns
            extra_cols = [col for col in header_normalized if col not in required_normalized]
            if extra_cols:
                return False, f"Extra columns found: {', '.join(extra_cols)}. Only {', '.join(CSVValidator.REQUIRED_COLUMNS)} are allowed."
                
            return True, "Valid CSV structure"
            
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}"
    
    @staticmethod
    def parse_date(date_str: str) -> Optional[pd.Timestamp]:
        """Parse date string in various formats."""
        if pd.isna(date_str) or str(date_str).strip() == '':
            return None
            
        date_str = str(date_str).strip()
        
        # Try different date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d/%m/%Y']
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
                
        return None
    
    @staticmethod
    def load_and_validate_csv(filepath: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Load CSV and validate data."""
        errors = []
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            if df.empty:
                return pd.DataFrame(columns=CSVValidator.REQUIRED_COLUMNS), []
            
            # Normalize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Parse dates
            for idx, row in df.iterrows():
                start_date = CSVValidator.parse_date(row['start'])
                end_date = CSVValidator.parse_date(row['end'])
                
                if start_date is None:
                    errors.append(f"Row {idx + 2}: Invalid start date '{row['start']}'")
                    continue
                    
                if end_date is None:
                    errors.append(f"Row {idx + 2}: Invalid end date '{row['end']}'")
                    continue
                    
                if start_date >= end_date:
                    errors.append(f"Row {idx + 2}: End date must be after start date")
                    continue
                    
                df.at[idx, 'start'] = start_date
                df.at[idx, 'end'] = end_date
            
            # Remove rows with errors
            df = df.dropna(subset=['start', 'end'])
            
            # Ensure proper data types
            df['name'] = df['name'].astype(str)
            df['owner'] = df['owner'].astype(str)
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
            
            return df, errors
            
        except Exception as e:
            errors.append(f"Error loading CSV: {str(e)}")
            return None, errors

class TaskEditDialog:
    """Dialog for editing task details."""
    
    @staticmethod
    def show_edit_dialog(task_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Show task edit dialog."""
        if task_data is None:
            task_data = {'name': '', 'start': '', 'end': '', 'owner': ''}
        
        layout = [
            [sg.Text('Task Name:'), sg.Input(task_data.get('name', ''), key='name', size=(30, 1))],
            [sg.Text('Start Date:'), sg.Input(task_data.get('start', ''), key='start', size=(20, 1)), 
             sg.Text('(YYYY-MM-DD)')],
            [sg.Text('End Date:'), sg.Input(task_data.get('end', ''), key='end', size=(20, 1)), 
             sg.Text('(YYYY-MM-DD)')],
            [sg.Text('Owner:'), sg.Input(task_data.get('owner', ''), key='owner', size=(30, 1))],
            [sg.Button('Save'), sg.Button('Cancel')]
        ]
        
        window = sg.Window('Edit Task', layout, modal=True, finalize=True)
        
        while True:
            event, values = window.read()
            
            if event in (sg.WIN_CLOSED, 'Cancel'):
                window.close()
                return None
                
            if event == 'Save':
                # Validate inputs
                errors = []
                
                if not values['name'].strip():
                    errors.append('Task name is required')
                    
                start_date = CSVValidator.parse_date(values['start'])
                if start_date is None:
                    errors.append('Invalid start date format')
                    
                end_date = CSVValidator.parse_date(values['end'])
                if end_date is None:
                    errors.append('Invalid end date format')
                    
                if start_date and end_date and start_date >= end_date:
                    errors.append('End date must be after start date')
                    
                if not values['owner'].strip():
                    errors.append('Owner is required')
                
                if errors:
                    sg.popup_error('Validation Errors:\n' + '\n'.join(errors))
                    continue
                
                result = {
                    'name': values['name'].strip(),
                    'start': start_date,
                    'end': end_date,
                    'owner': values['owner'].strip()
                }
                
                window.close()
                return result

class GanttApp:
    """Main application class."""
    
    def __init__(self):
        self.df = pd.DataFrame(columns=['name', 'start', 'end', 'owner'])
        self.current_file = None
        self.has_unsaved_changes = False
        self.chart = GanttChart()
        self.undo_manager = UndoRedoManager()
        
        # Theme
        sg.theme('LightBlue3')
        
        # Create main window
        self.window = self.create_main_window()
        self.setup_canvas()
        
        logging.info("Gantt Chart Application started")
    
    def create_main_window(self):
        """Create the main application window."""
        # Menu bar
        menu_def = [
            ['File', ['Load CSV::load', 'Save::save', 'Save As::save_as', '---', 'Exit::exit']],
            ['Edit', ['Add Task::add_task', 'Delete Task::delete_task', '---', 
                     'Undo::undo', 'Redo::redo']],
            ['View', ['Zoom In::zoom_in', 'Zoom Out::zoom_out', 'Reset View::reset_view']],
            ['Help', ['About::about']]
        ]
        
        # Main layout
        left_column = [
            [sg.Button('Load CSV', key='load_csv', size=(15, 1))],
            [sg.Button('Add Task', key='add_task', size=(15, 1))],
            [sg.Button('Delete Task', key='delete_task', size=(15, 1))],
            [sg.Button('Save CSV', key='save_csv', size=(15, 1))],
            [sg.HSeparator()],
            [sg.Text('Tasks:', font=('Arial', 12, 'bold'))],
            [sg.Listbox(values=[], key='task_list', size=(25, 20), 
                       enable_events=True, bind_return_key=True)]
        ]
        
        right_column = [
            [sg.Canvas(key='canvas', size=(900, 600), background_color='white')]
        ]
        
        layout = [
            [sg.Menu(menu_def)],
            [sg.Column(left_column, vertical_alignment='top'), 
             sg.VSeparator(),
             sg.Column(right_column, expand_x=True, expand_y=True)],
            [sg.StatusBar('Ready', key='status_bar', size=(100, 1))]
        ]
        
        return sg.Window('Gantt Chart Manager', layout, 
                        resizable=True, finalize=True, 
                        size=(1300, 700), 
                        enable_close_attempted_event=True)
    
    def setup_canvas(self):
        """Setup the matplotlib canvas."""
        self.chart.create_figure()
        self.chart.canvas = FigureCanvasTkAgg(self.chart.fig, self.window['canvas'].TKCanvas)
        self.chart.canvas.draw()
        self.chart.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        # Connect mouse events
        self.chart.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.chart.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.chart.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
    
    def update_task_list(self):
        """Update the task list display."""
        if self.df.empty:
            self.window['task_list'].update([])
        else:
            task_names = [f"{i+1}. {row['name']} ({row['owner']})" 
                         for i, (_, row) in enumerate(self.df.iterrows())]
            self.window['task_list'].update(task_names)
    
    def update_status(self, message: str):
        """Update status bar."""
        self.window['status_bar'].update(message)
        self.window.refresh()
    
    def mark_unsaved_changes(self):
        """Mark that there are unsaved changes."""
        self.has_unsaved_changes = True
        title = self.window.Title
        if not title.endswith('*'):
            self.window.TKroot.title(title + '*')
    
    def clear_unsaved_changes(self):
        """Clear unsaved changes flag."""
        self.has_unsaved_changes = False
        title = self.window.Title
        if title.endswith('*'):
            self.window.TKroot.title(title[:-1])
    
    def check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt user."""
        if not self.has_unsaved_changes:
            return True
            
        choice = sg.popup_yes_no_cancel('You have unsaved changes. Save before continuing?',
                                       title='Unsaved Changes')
        if choice == 'Yes':
            return self.save_csv()
        elif choice == 'No':
            return True
        else:  # Cancel
            return False
    
    def load_csv(self, filepath: str = None):
        """Load CSV file."""
        if not self.check_unsaved_changes():
            return False
            
        if filepath is None:
            filepath = sg.popup_get_file('Select CSV file', 
                                       file_types=(('CSV Files', '*.csv'),))
            if not filepath:
                return False
        
        # Validate CSV structure
        is_valid, message = CSVValidator.validate_csv_structure(filepath)
        if not is_valid:
            sg.popup_error(f'Invalid CSV file:\n{message}')
            return False
        
        # Load and validate data
        df, errors = CSVValidator.load_and_validate_csv(filepath)
        
        if df is None:
            sg.popup_error('Failed to load CSV file:\n' + '\n'.join(errors))
            return False
        
        if errors:
            sg.popup_scrolled('Data validation warnings:\n' + '\n'.join(errors),
                            title='Validation Warnings', size=(60, 20))
        
        self.df = df
        self.current_file = filepath
        self.clear_unsaved_changes()
        self.undo_manager = UndoRedoManager()  # Reset undo history
        self.undo_manager.save_state(self.df)
        
        self.update_task_list()
        self.chart.plot_gantt(self.df)
        self.update_status(f'Loaded {len(self.df)} tasks from {os.path.basename(filepath)}')
        
        logging.info(f"Loaded CSV file: {filepath}")
        return True
    
    def save_csv(self, filepath: str = None) -> bool:
        """Save CSV file."""
        if self.df.empty:
            sg.popup('No data to save')
            return False
        
        if filepath is None:
            if self.current_file:
                choice = sg.popup_yes_no(f'Overwrite {os.path.basename(self.current_file)}?')
                if choice == 'Yes':
                    filepath = self.current_file
                else:
                    return False
            else:
                filepath = sg.popup_get_file('Save CSV as', save_as=True,
                                           file_types=(('CSV Files', '*.csv'),))
                if not filepath:
                    return False
        
        try:
            # Format dates for saving
            df_save = self.df.copy()
            df_save['start'] = df_save['start'].dt.strftime('%Y-%m-%d')
            df_save['end'] = df_save['end'].dt.strftime('%Y-%m-%d')
            
            df_save.to_csv(filepath, index=False)
            
            self.current_file = filepath
            self.clear_unsaved_changes()
            self.update_status(f'Saved to {os.path.basename(filepath)}')
            
            logging.info(f"Saved CSV file: {filepath}")
            return True
            
        except Exception as e:
            sg.popup_error(f'Error saving file:\n{str(e)}')
            logging.error(f"Error saving CSV: {str(e)}")
            return False
    
    def add_task(self):
        """Add a new task."""
        task_data = TaskEditDialog.show_edit_dialog()
        if task_data is None:
            return
        
        self.undo_manager.save_state(self.df)
        
        # Add new task
        new_row = pd.DataFrame([task_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        self.mark_unsaved_changes()
        self.update_task_list()
        self.chart.plot_gantt(self.df)
        self.update_status(f'Added task: {task_data["name"]}')
        
        logging.info(f"Added task: {task_data['name']}")
    
    def delete_task(self):
        """Delete selected task."""
        selection = self.window['task_list'].get()
        if not selection:
            sg.popup('Please select a task to delete')
            return
        
        # Get selected index
        selected_idx = self.window['task_list'].get_indexes()[0]
        task_name = self.df.iloc[selected_idx]['name']
        
        choice = sg.popup_yes_no(f'Delete task "{task_name}"?')
        if choice != 'Yes':
            return
        
        self.undo_manager.save_state(self.df)
        
        # Delete task
        self.df = self.df.drop(self.df.index[selected_idx]).reset_index(drop=True)
        
        self.mark_unsaved_changes()
        self.update_task_list()
        self.chart.plot_gantt(self.df)
        self.update_status(f'Deleted task: {task_name}')
        
        logging.info(f"Deleted task: {task_name}")
    
    def edit_task(self, task_idx: int):
        """Edit existing task."""
        if task_idx >= len(self.df):
            return
        
        task_row = self.df.iloc[task_idx]
        task_data = {
            'name': task_row['name'],
            'start': task_row['start'].strftime('%Y-%m-%d'),
            'end': task_row['end'].strftime('%Y-%m-%d'),
            'owner': task_row['owner']
        }
        
        new_data = TaskEditDialog.show_edit_dialog(task_data)
        if new_data is None:
            return
        
        self.undo_manager.save_state(self.df)
        
        # Update task
        for key, value in new_data.items():
            self.df.at[task_idx, key] = value
        
        self.mark_unsaved_changes()
        self.update_task_list()
        self.chart.plot_gantt(self.df)
        self.update_status(f'Updated task: {new_data["name"]}')
        
        logging.info(f"Updated task: {new_data['name']}")
    
    def undo(self):
        """Undo last action."""
        df = self.undo_manager.undo()
        if df is not None:
            self.df = df
            self.mark_unsaved_changes()
            self.update_task_list()
            self.chart.plot_gantt(self.df)
            self.update_status('Undo performed')
    
    def redo(self):
        """Redo last undone action."""
        df = self.undo_manager.redo()
        if df is not None:
            self.df = df
            self.mark_unsaved_changes()
            self.update_task_list()
            self.chart.plot_gantt(self.df)
            self.update_status('Redo performed')
    
    def on_mouse_press(self, event):
        """Handle mouse press events on chart."""
        if event.inaxes != self.chart.ax or self.df.empty:
            return
        
        # Find clicked task
        if event.ydata is not None:
            task_idx = int(round(event.ydata))
            if 0 <= task_idx < len(self.df):
                self.chart.selected_task = task_idx
                
                # Double-click to edit
                if event.dblclick:
                    self.edit_task(task_idx)
    
    def on_mouse_release(self, event):
        """Handle mouse release events."""
        self.chart.dragging = False
        self.chart.drag_task = None
        self.chart.drag_handle = None
    
    def on_mouse_motion(self, event):
        """Handle mouse motion events."""
        # This is simplified - full drag implementation would be more complex
        pass
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
Gantt Chart Desktop Application
Version 1.0.0

A standalone desktop application for creating and managing 
Gantt charts from CSV files.

Features:
• Load and save CSV files
• Interactive Gantt chart visualization
• Task editing and management
• Undo/Redo functionality
• Cross-platform compatibility

Author: Claude AI Assistant
License: MIT License

Dependencies:
• PySimpleGUI
• Matplotlib
• Pandas
        """
        sg.popup_scrolled(about_text, title='About Gantt Chart App', size=(50, 20))
    
    def handle_file_drop(self, filepath: str):
        """Handle file drop event."""
        if filepath.lower().endswith('.csv'):
            self.load_csv(filepath)
        else:
            sg.popup_error('Please drop a CSV file')
    
    def run(self):
        """Main application loop."""
        # Initial chart display
        self.chart.plot_gantt(self.df)
        
        while True:
            event, values = self.window.read()
            
            if event == sg.WIN_CLOSED or event == 'exit':
                if self.check_unsaved_changes():
                    break
                continue
            
            # Handle close attempt
            if event == '-WINDOW CLOSE ATTEMPTED-':
                if self.check_unsaved_changes():
                    break
                continue
            
            # File menu
            if event in ('load', 'load_csv'):
                self.load_csv()
            elif event in ('save', 'save_csv'):
                self.save_csv()
            elif event == 'save_as':
                self.save_csv(None)
            
            # Edit menu
            elif event in ('add_task',):
                self.add_task()
            elif event in ('delete_task',):
                self.delete_task()
            elif event == 'undo':
                self.undo()
            elif event == 'redo':
                self.redo()
            
            # View menu
            elif event == 'zoom_in':
                # Simplified zoom - would need more complex implementation
                self.update_status('Zoom in (feature in development)')
            elif event == 'zoom_out':
                self.update_status('Zoom out (feature in development)')
            elif event == 'reset_view':
                self.chart.plot_gantt(self.df)
                self.update_status('View reset')
            
            # Help menu
            elif event == 'about':
                self.show_about()
            
            # Task list events
            elif event == 'task_list':
                if values['task_list']:
                    selected_idx = self.window['task_list'].get_indexes()[0]
                    # Could highlight selected task in chart
            
            # Handle keyboard shortcuts
            elif event.startswith('Control_L'):
                key = event.split(':')[1] if ':' in event else ''
                if key == 'o':
                    self.load_csv()
                elif key == 's':
                    self.save_csv()
                elif key == 'z':
                    self.undo()
                elif key == 'y':
                    self.redo()
                elif key == 'q':
                    if self.check_unsaved_changes():
                        break
        
        self.window.close()
        logging.info("Application closed")

def main():
    """Main entry point."""
    try:
        app = GanttApp()
        app.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        sg.popup_error(f'Application error:\n{str(e)}')

if __name__ == '__main__':
    main()