# functions.py

def analyze_quiz_failure(student_id, quiz_id, quiz_scores_df, videos_df, video_interactions_df, quizzes_df, score_threshold=70, watch_threshold=60):
    student_score = quiz_scores_df[
        (quiz_scores_df["student_id"] == student_id) &
        (quiz_scores_df["quiz_id"] == quiz_id)
    ]

    if student_score.empty:
        return f"No record of {student_id} taking {quiz_id}."

    score = student_score.iloc[0]["score"]
    if score >= score_threshold:
        return f"{student_id} passed {quiz_id} with a score of {score}."

    quiz_topic = quizzes_df[quizzes_df["quiz_id"] == quiz_id]["topic"].values[0]
    topic_videos = videos_df[videos_df["topic"] == quiz_topic]["video_id"].tolist()

    watched = video_interactions_df[
        (video_interactions_df["student_id"] == student_id) &
        (video_interactions_df["video_id"].isin(topic_videos))
    ]

    result = [f"{student_id} failed {quiz_id} (Score: {score})."]

    if watched.empty:
        result.append(f"They did not watch any videos on the topic '{quiz_topic}' before the quiz.")
    else:
        for _, row in watched.iterrows():
            vid = row["video_id"]
            watch_time = row["watch_time"]
            title = videos_df[videos_df["video_id"] == vid]["title"].values[0]
            result.append(f"They watched '{title}' (ID: {vid}) for {watch_time} seconds.")

            if watch_time < watch_threshold:
                result.append("⛔ They watched this video for too short a time to be helpful.")

    passing_students = quiz_scores_df[
        (quiz_scores_df["quiz_id"] == quiz_id) &
        (quiz_scores_df["score"] >= score_threshold)
    ]["student_id"].unique()

    successful_views = video_interactions_df[
        (video_interactions_df["student_id"].isin(passing_students)) &
        (video_interactions_df["video_id"].isin(topic_videos))
    ]

    if not successful_views.empty:
        top_video = successful_views.groupby("video_id")["watch_time"].mean().sort_values(ascending=False).index[0]
        top_video_title = videos_df[videos_df["video_id"] == top_video]["title"].values[0]
        result.append(f"✅ Suggestion: Students who passed watched '{top_video_title}' (ID: {top_video}) more effectively.")
    else:
        result.append("⚠️ No helpful video data from students who passed this quiz.")

    return "\n".join(result)

# --- Function to Get Fastest Student on a Quiz ---
def fastest_student_on_quiz(graph, quiz_id):
    times = [
        (u, data["time_taken"])
        for u, v, data in graph.edges(data=True)
        if data.get("relation") == "took" and v == quiz_id
    ]
    return min(times, key=lambda x: x[1]) if times else None

# --- Function to Get Slowest Student on a Quiz ---
def slowest_student_on_quiz(graph, quiz_id):
    times = [
        (u, data["time_taken"])
        for u, v, data in graph.edges(data=True)
        if data.get("relation") == "took" and v == quiz_id
    ]
    return max(times, key=lambda x: x[1]) if times else None

# --- Function to Flag Potential Cheaters ---
def flag_potential_cheaters(graph, quiz_id, threshold=0.3):
    times = [
        data["time_taken"]
        for u, v, data in graph.edges(data=True)
        if data.get("relation") == "took" and v == quiz_id
    ]
    if not times:
        return []
    avg_time = sum(times) / len(times)
    flagged = [
        u for u, v, data in graph.edges(data=True)
        if data.get("relation") == "took" and v == quiz_id and data["time_taken"] < avg_time * threshold and data["score"] >= 90
    ]
    return flagged

# --- Function to Get Most Effective Videos ---
def get_effective_videos(video_interactions_df, quiz_scores_df, threshold=80):
    high_scores = quiz_scores_df[quiz_scores_df["score"] >= threshold]["student_id"].unique()
    good_views = video_interactions_df[video_interactions_df["student_id"].isin(high_scores)]
    if good_views.empty:
        return []
    return good_views.groupby("video_id")["watch_time"].mean().sort_values(ascending=False).index.tolist()

def average_time_per_question(quiz_id, question_times_df, quiz_questions_df):
    questions = quiz_questions_df[quiz_questions_df["quiz_id"] == quiz_id]["question_id"].tolist()
    question_avgs = {}
    
    for qid in questions:
        times = question_times_df[question_times_df["question_id"] == qid]["time_on_question"]
        if not times.empty:
            question_avgs[qid] = round(times.mean(), 2)
        else:
            question_avgs[qid] = 0  # No data available
    
    return question_avgs

def student_learning_profile(student_id, quiz_scores_df, video_interactions_df, question_times_df, quizzes_df):
    profile = {}

    # Overall quiz performance
    student_scores = quiz_scores_df[quiz_scores_df["student_id"] == student_id]
    if not student_scores.empty:
        profile["average_score"] = round(student_scores["score"].mean(), 2)
        profile["quizzes_taken"] = student_scores["quiz_id"].tolist()
    else:
        profile["average_score"] = None
        profile["quizzes_taken"] = []

    # Watch behavior
    student_videos = video_interactions_df[video_interactions_df["student_id"] == student_id]
    if not student_videos.empty:
        profile["average_watch_time"] = round(student_videos["watch_time"].mean(), 2)
        profile["videos_watched"] = student_videos["video_id"].tolist()
    else:
        profile["average_watch_time"] = 0
        profile["videos_watched"] = []

    # Question response timing
    student_questions = question_times_df[question_times_df["student_id"] == student_id]
    if not student_questions.empty:
        profile["avg_time_per_question"] = round(student_questions["time_on_question"].mean(), 2)
    else:
        profile["avg_time_per_question"] = 0

    return profile

def identify_struggling_students(quiz_scores_df, video_interactions_df, threshold_score=60, watch_threshold=60):
    # Students with low average quiz scores
    avg_scores = quiz_scores_df.groupby("student_id")["score"].mean()
    low_performers = avg_scores[avg_scores < threshold_score].index.tolist()

    # Students who don't watch videos long enough
    avg_watch = video_interactions_df.groupby("student_id")["watch_time"].mean()
    low_watchers = avg_watch[avg_watch < watch_threshold].index.tolist()

    # Combine both factors
    struggling = set(low_performers) | set(low_watchers)

    return {
        "struggling_due_to_low_scores": low_performers,
        "struggling_due_to_low_engagement": low_watchers,
        "overall_struggling_students": list(struggling)
    }

def recommend_video_for_student(student_id, quiz_id, quiz_scores_df, quizzes_df, videos_df, video_interactions_df, score_threshold=70):
    # Topic of the quiz
    quiz_topic = quizzes_df[quizzes_df["quiz_id"] == quiz_id]["topic"].values[0]

    # Videos related to this topic
    topic_videos = videos_df[videos_df["topic"] == quiz_topic]["video_id"].tolist()

    # Videos watched by this student
    watched = video_interactions_df[
        (video_interactions_df["student_id"] == student_id) &
        (video_interactions_df["video_id"].isin(topic_videos))
    ]["video_id"].tolist()

    # Students who did well
    successful_students = quiz_scores_df[
        (quiz_scores_df["quiz_id"] == quiz_id) &
        (quiz_scores_df["score"] >= score_threshold)
    ]["student_id"].unique()

    # Videos they watched on this topic
    successful_views = video_interactions_df[
        (video_interactions_df["student_id"].isin(successful_students)) &
        (video_interactions_df["video_id"].isin(topic_videos))
    ]

    if successful_views.empty:
        return f"No video data available from successful students for {quiz_id}."

    # Recommend video with highest average watch time that student hasn’t watched yet
    video_scores = successful_views.groupby("video_id")["watch_time"].mean().sort_values(ascending=False)
    for vid in video_scores.index:
        if vid not in watched:
            title = videos_df[videos_df["video_id"] == vid]["title"].values[0]
            return f"✅ Recommend: '{title}' (ID: {vid}) based on success patterns from other students."

    return "⚠️ No new videos to recommend — student has already watched all top videos."
