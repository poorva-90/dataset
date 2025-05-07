import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from datetime import datetime
import os
import json
from kagglehub import dataset_download

# Authenticate with Kaggle using kaggle.json
kaggle_json_path = "kaggle.json"  # File must be in the same directory
with open(kaggle_json_path, 'r') as f:
    creds = json.load(f)
os.environ["KAGGLE_USERNAME"] = creds["username"]
os.environ["KAGGLE_KEY"] = creds["key"]

# Download dataset
path = dataset_download("wcukierski/enron-email-dataset")
df = pd.read_csv(path / "emails.csv")
print("Dataset loaded successfully!")

def parse_email_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    
    date_formats = [
        '%a, %d %b %Y %H:%M:%S %z',  # Tue, 15 May 2001 08:30:00 -0700
        '%a, %d %b %Y %H:%M:%S %Z',   # Tue, 15 May 2001 08:30:00 PST
        '%d %b %Y %H:%M:%S %z',       # 15 May 2001 08:30:00 -0700
        '%m/%d/%Y %H:%M:%S',          # 05/15/2001 08:30:00
        '%Y-%m-%d %H:%M:%S'           # 2001-05-15 08:30:00
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    return pd.NaT

def extract_email_info(message):
    info = {
        'from': None,
        'to': None,
        'subject': None,
        'date': None,
        'body': None,
        'has_attachment': False
    }
    
   
    from_match = re.search(r'From:\s*(.+)', message)
    to_match = re.search(r'To:\s*(.+)', message)
    subject_match = re.search(r'Subject:\s*(.+)', message)
    date_match = re.search(r'Date:\s*(.+)', message)
    
    if from_match: info['from'] = from_match.group(1).strip()
    if to_match: info['to'] = to_match.group(1).strip()
    if subject_match: info['subject'] = subject_match.group(1).strip()
    
   
    if date_match:
        info['date'] = parse_email_date(date_match.group(1))
    

    parts = message.split('\n\n', 1)
    info['body'] = parts[1] if len(parts) > 1 else None
    info['has_attachment'] = bool(re.search(r'X-FileName:', message))
    
    return info


print("\nExtracting email metadata...")
email_info = df['message'].apply(extract_email_info)
df = pd.concat([df, pd.DataFrame(list(email_info))], axis=1)

df['date'] = pd.to_datetime(df['date'], errors='coerce')

print("\n1. Dataset structure:")
print(f"Total emails: {len(df):,}")
print("Columns:", df.columns.tolist())
print(f"Valid dates: {df['date'].notna().sum():,} of {len(df):,}")

unique_senders = df['from'].nunique()
print(f"\n2. Unique senders: {unique_senders:,}")

top_senders = df['from'].value_counts().head(5)
print("\n3. Top 5 senders:")
print(top_senders)

all_recipients = df['to'].str.split(',').explode().str.strip().dropna()
top_recipients = all_recipients.value_counts().head(5)
print("\n4. Top 5 recipients:")
print(top_recipients)

avg_recipients = df['to'].str.split(',').str.len().mean()
print(f"\n5. Avg recipients/email: {avg_recipients:.2f}")

df['body_length'] = df['body'].str.len()
print("\n6. Email length stats (chars):")
print(df['body_length'].describe().apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN"))

valid_dates = df[df['date'].notna()]
if not valid_dates.empty:
    valid_dates['day_of_week'] = valid_dates['date'].dt.day_name()
    busiest_day = valid_dates['day_of_week'].value_counts().idxmax()
    print(f"\n7. Busiest day: {busiest_day}")
else:
    print("\n7. No valid dates to determine busiest day")

if not valid_dates.empty:
    valid_dates['hour'] = valid_dates['date'].dt.hour
    busiest_hour = valid_dates['hour'].value_counts().idxmax()
    print(f"\n8. Busiest hour: {busiest_hour}:00")
else:
    print("\n8. No valid dates to determine busiest hour")

email_lengths = np.array(df['body_length'].dropna())
if len(email_lengths) > 0:
    print("\n9. Numpy email length stats:")
    print(f"Min: {np.min(email_lengths):,}")
    print(f"Max: {np.max(email_lengths):,}")
    print(f"Mean: {np.mean(email_lengths):,.0f}")
    print(f"Median: {np.median(email_lengths):,.0f}")
else:
    print("\n9. No email body length data available")

attachment_percent = df['has_attachment'].mean() * 100
print(f"\n10. Emails with attachments: {attachment_percent:.2f}%")

if df['subject'].notna().any():
    subjects = ' '.join(df['subject'].dropna().str.lower())
    words = re.findall(r'\b\w{4,}\b', subjects)
    common_words = Counter(words).most_common(10)
    print("\n11. Top 10 subject words:")
    print(common_words)
else:
    print("\n11. No subject data available")

if df['body_length'].notna().any():
    longest_idx = df['body_length'].idxmax()
    longest = df.loc[longest_idx]
    print("\n12. Longest email:")
    print(f"From: {longest['from']}")
    print(f"Subject: {longest['subject']}")
    print(f"Length: {longest['body_length']:,} chars")
else:
    print("\n12. No email body data available")

if not valid_dates.empty:
    emails_per_month = valid_dates.set_index('date').resample('M').size()
    print("\n13. Emails per month:")
    print(emails_per_month.head())
else:
    print("\n13. No valid dates for monthly analysis")

if df['from'].notna().any():
    df['sender_domain'] = df['from'].str.extract(r'@([\w.-]+)')
    top_domains = df['sender_domain'].value_counts().head(5)
    print("\n14. Top sender domains:")
    print(top_domains)
else:
    print("\n14. No sender data available")

if df['to'].notna().any():
    mass_emails = df[df['to'].str.split(',').str.len() > 10]
    print(f"\n15. Mass emails (>10 recipients): {len(mass_emails):,}")
else:
    print("\n15. No recipient data available")

if len(valid_dates) > 1:
    valid_dates_sorted = valid_dates.sort_values('date')
    time_diffs = valid_dates_sorted['date'].diff().dt.total_seconds()
    avg_response_hours = np.mean(time_diffs[time_diffs > 0]) / 3600
    print(f"\n16. Avg response time: {avg_response_hours:.2f} hours")
else:
    print("\n16. Not enough valid dates for response time analysis")

if df['subject'].notna().any():
    active_thread = df['subject'].value_counts().idxmax()
    count = df['subject'].value_counts().max()
    print(f"\n17. Most active thread: '{active_thread}' ({count:,} emails)")
else:
    print("\n17. No subject data available")

if not valid_dates.empty and 'day_of_week' in valid_dates.columns:
    pivot = pd.pivot_table(valid_dates, 
                          values='body', 
                          index='hour', 
                          columns='day_of_week', 
                          aggfunc='count', 
                          fill_value=0)
    print("\n18. Emails by hour/day:")
    print(pivot.head())
else:
    print("\n18. No valid date/day data available")

if df['sender_domain'].notna().any() and df['to'].notna().any():
    internal = df['sender_domain'].eq('enron.com') & df['to'].str.contains('enron.com', na=False)
    internal_pct = internal.mean() * 100
    print(f"\n19. Internal emails: {internal_pct:.2f}%")
else:
    print("\n19. Missing data for internal email analysis")

if df['body'].notna().any():
    print("\n20. Calculating top bigrams...")
    sample_size = min(1000, len(df))
    text_sample = ' '.join(df['body'].dropna().sample(sample_size, random_state=42).str.lower())
    words = re.findall(r'\b\w{3,}\b', text_sample)  # words with 3+ letters
    bigrams = list(zip(words, words[1:]))
    bigram_counts = Counter(bigrams)
    print("Top 10 bigrams:")
    for bigram, count in bigram_counts.most_common(10):
        print(f"{' '.join(bigram)}: {count}")
else:
    print("\n20. No email body data available")
