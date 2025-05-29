import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from graph_visualizer import visualize_student_subgraph
from functions import (
    analyze_quiz_failure,
    fastest_student_on_quiz,
    slowest_student_on_quiz,
    flag_potential_cheaters,
    get_effective_videos
)

# Only call this when you actually want to see the graph:
# visualize_student_subgraph(G, "S014")


# #Make sure the files are in directory
# print("Current working directory:", os.getcwd())
# print("Files in directory:", os.listdir())

# Load CSV files
students_df = pd.read_csv("students.csv")
videos_df = pd.read_csv("videos.csv")
quizzes_df = pd.read_csv("quizzes.csv")
quiz_questions_df = pd.read_csv("quiz_questions.csv")
video_interactions_df = pd.read_csv("student_video_interactions.csv")
quiz_scores_df = pd.read_csv("student_quiz_scores.csv")
question_times_df = pd.read_csv("student_question_times.csv")

# Create a directed graph
G = nx.DiGraph()

# --- Add student nodes ---
for _, row in students_df.iterrows():
    G.add_node(row["student_id"], type="student", name=row["name"], major=row["major"])

# --- Add video nodes ---
for _, row in videos_df.iterrows():
    G.add_node(row["video_id"], type="video", title=row["title"], topic=row["topic"], difficulty=row["difficulty"])

# --- Add quiz nodes ---
for _, row in quizzes_df.iterrows():
    G.add_node(row["quiz_id"], type="quiz", topic=row["topic"], difficulty=row["difficulty"])

# --- Add question nodes and connect them to their quiz ---
for _, row in quiz_questions_df.iterrows():
    G.add_node(row["question_id"], type="question", topic=row["topic"])
    G.add_edge(row["quiz_id"], row["question_id"], relation="contains")

# --- Add edges for students watching videos ---
for _, row in video_interactions_df.iterrows():
    G.add_edge(row["student_id"], row["video_id"], relation="watched",
               watch_time=row["watch_time"], rewatched=row["rewatched"], completed=row["completed"])

# --- Add edges for students taking quizzes ---
for _, row in quiz_scores_df.iterrows():
    G.add_edge(row["student_id"], row["quiz_id"], relation="took",
               score=row["score"], time_taken=row["time_taken"])

# --- Add edges for students spending time on quiz questions ---
for _, row in question_times_df.iterrows():
    G.add_edge(row["student_id"], row["question_id"], relation="spent_time_on",
               time_on_question=row["time_on_question"])


# Example: visualize one student
# visualize_student_subgraph(G, "S014")

# Summary of the graph
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Node types:", set(nx.get_node_attributes(G, 'type').values()))


result = analyze_quiz_failure("S014", "Q002", quiz_scores_df, videos_df, video_interactions_df, quizzes_df)
print(result)

