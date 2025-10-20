import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def extract_comments(f):
    """
    Extract all comments from an ADI file with time information.
    
    Parameters
    ----------
    f : adi.File
        An opened ADI file object from adi.read_file()
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all comments with columns:
        - text: Comment text
        - tick_position: Position in ticks within the record
        - channel_: Channel number (-1 if not channel-specific)
        - id: Comment ID
        - tick_dt: Time between ticks (sampling period)
        - time: Time in seconds from record start
        - record_id: Record number (1-based)
        - datetime: Datetime of the comment in local time of the recording device
        - seconds_since_midnight: Seconds since midnight of the record's day
        - record_start_datetime: Datetime when the record started
    
    Examples
    --------
    >>> f = adi.read_file('data.adicht')
    >>> comments_df = extract_comments(f)
    >>> print(comments_df[['text', 'datetime', 'record_id']].head())
    """
    comments_with_context = []
    for record in f.records:
        record_start = record.record_time.rec_datetime
        midnight = record_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        for comment in record.comments:
            # Add absolute time information to comment
            comment_dict = comment.__dict__.copy()
            comment_dict['record_id'] = record.id
            comment_dict['datetime'] = record_start + timedelta(seconds=comment.time)
            comment_dict['seconds_since_midnight'] = (comment_dict['datetime'] - midnight).total_seconds()
            comment_dict['record_start_datetime'] = record_start
            comments_with_context.append(comment_dict)

    return pd.DataFrame(comments_with_context)

def extract_channels(f):
    """
    Extract channel information from an ADI file.
    
    Parameters
    ----------
    f : adi.File
        An opened ADI file object from adi.read_file()
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing channel information with columns:
        - id: Channel ID (1-based)
        - name: Channel name
        - units: Channel units (first unique value if multiple)
        - dt_s: Sampling period in seconds (first unique value if multiple)
        - frequency_hz: Sampling frequency in Hz (first unique value if multiple)
    
    Notes
    -----
    If units or sampling rates change during the experiment, only the first
    unique value is returned. This may cause issues in downstream functions
    if not handled properly. All unique values are preserved in the original
    channel objects if needed.
    
    Examples
    --------
    >>> f = adi.read_file('data.adicht')
    >>> channels_df = extract_channels(f)
    >>> print(channels_df[['id', 'name', 'units', 'frequency_hz']])
    """
    channels = []
    for channel in f.channels:
        channel_dict = dict()
        channel_dict['id'] = channel.id
        channel_dict['name'] = channel.name
        # May break downstream functions if user changes units during experiment, but all unit settings will be preserved
        channel_dict['units'] = list(set(channel.units))[0]
        channel_dict['dt_s'] = list(set(channel.dt))[0]
        channel_dict['frequency_hz'] = list(set(channel.fs))[0]
        channels.append(channel_dict)

    return pd.DataFrame(channels)

def convert_time(time, f=None):
    """
    Convert various time formats to datetime objects.
    
    Parameters
    ----------
    time : datetime, float, int, or np.ndarray
        Time to convert. Can be:
        - datetime object (returned as-is)
        - float/int representing seconds since midnight (requires f)
        - np.ndarray containing datetime (for R/reticulate compatibility)
    f : adi.File, optional
        ADI file object used to get reference date for seconds-since-midnight
        conversion. Only required when time is numeric.
    
    Returns
    -------
    datetime
        Converted datetime object
    
    Raises
    ------
    ValueError
        If time is numeric but f is not provided
    """
    if isinstance(time, (int, float)):
        if f is None:
            raise ValueError("File object 'f' is required to convert numeric time values")
        # Assume int is seconds since midnight - use date from first record
        first_record_date = f.records[0].record_time.rec_datetime.date()
        midnight = datetime.combine(first_record_date, datetime.min.time())
        event_time = midnight + timedelta(seconds=time)
    elif isinstance(time, np.ndarray):
        # Reticulate may pass ndarray in dict -- convert
        event_time = pd.to_datetime(time[0])
    elif isinstance(time, datetime):
        event_time = time
    else:
        raise TypeError(f"Unsupported time type: {type(time)}")
    
    return event_time

def create_window(event_datetime, seconds_before=300, seconds_after=300, f=None):
    """
    Create a time window around an event.
    
    Parameters
    ----------
    event_datetime : datetime, float, int, or np.ndarray
        The center time of the window. Can be any format accepted by convert_time().
    seconds_before : float, optional
        Seconds before event_datetime to include in window. Default is 300 (5 minutes).
    seconds_after : float, optional
        Seconds after event_datetime to include in window. Default is 300 (5 minutes).
    f : adi.File, optional
        ADI file object. Required only if event_datetime is numeric (seconds since midnight).
    
    Returns
    -------
    tuple of datetime
        (window_start, window_end) defining the time window
    
    Examples
    --------
    >>> # With datetime object (no f needed)
    >>> event_time = datetime(2024, 1, 1, 12, 0, 0)
    >>> start, end = create_window(event_time)
    
    >>> # With numeric time (f required)
    >>> f = adi.read_file('data.adicht')
    >>> start, end = create_window(3600, f=f)  # 1:00:00 AM
    """
    event_datetime = convert_time(event_datetime, f)
    
    window_start = event_datetime - timedelta(seconds=seconds_before)
    window_end = event_datetime + timedelta(seconds=seconds_after)
    
    return window_start, window_end


def generate_timepoints(event_datetime, interval_s, n, f=None):
    """
    Generate a series of evenly-spaced timepoints starting from an event.
    
    Parameters
    ----------
    event_datetime : datetime, float, int, or np.ndarray
        The starting time. Can be any format accepted by convert_time().
    interval_s : float
        Time interval in seconds between consecutive timepoints.
    n : int
        Number of additional timepoints to generate after the starting time.
    f : adi.File, optional
        ADI file object. Required only if event_datetime is numeric (seconds since midnight).
    
    Returns
    -------
    list of datetime
        List containing n+1 timepoints: the event_datetime plus n additional
        timepoints spaced at interval_s seconds.
    
    Examples
    --------
    >>> # With datetime object
    >>> start = datetime(2024, 1, 1, 12, 0, 0)
    >>> timepoints = generate_timepoints(start, interval_s=30, n=10)
    
    >>> # With numeric time
    >>> f = adi.read_file('data.adicht')
    >>> timepoints = generate_timepoints(7200, interval_s=30, n=10, f=f)  # Start at 2 AM
    """
    event_datetime = convert_time(event_datetime, f)
    timepoints = [event_datetime]

    for _ in range(1, int(n)+1):
        timepoints.append(event_datetime + timedelta(seconds=interval_s * _))

    return timepoints

def find_records(f, event_datetimes):
    """
    Find which records contain data for a time window defined by event datetimes.
    
    Parameters
    ----------
    f : adi.File
        An opened ADI file object from adi.read_file()
    event_datetimes : list of datetime
        List of datetime objects. The function finds records that overlap with
        the time span from min(event_datetimes) to max(event_datetimes).
    
    Returns
    -------
    list of dict
        List of dictionaries, one per overlapping record, containing:
        - record_id: Record number (1-based)
        - record_start_datetime: When the record started
        - record_end_datetime: When the record ended  
        - start_sample: First sample index to extract (1-based)
        - end_sample: Last sample index to extract (1-based)
        - start_time_in_record: Time offset of first sample from record start (seconds)
        - end_time_in_record: Time offset of last sample from record start (seconds)
        - dt: Sampling period for this record (seconds)
    
    Notes
    -----
    This function is typically used internally to identify which records need
    to be accessed when extracting data that spans multiple recording sessions.
    Sample indices are 1-based to match the ADI library convention.
    
    Examples
    --------
    >>> f = adi.read_file('data.adicht')
    >>> # Find records containing data between two times
    >>> times = [datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 5, 0)]
    >>> records = find_records(f, times)
    >>> print(f"Data spans {len(records)} record(s)")
    """

    # Calculate window boundaries
    window_start = min(event_datetimes)
    window_end = max(event_datetimes)
    
    matching_records = []
    
    for record in f.records:
        record_start = record.record_time.rec_datetime
        record_duration = record.n_ticks * record.tick_dt  # duration in seconds
        record_end = record_start + timedelta(seconds=record_duration)
        
        # Check if record overlaps with window
        if record_start <= window_end and record_end >= window_start:
            # Calculate the overlap region in terms of sample indices
            # Time relative to record start. Don't exceed record time bounds (redundant).
            overlap_start_time = max(0, (window_start - record_start).total_seconds())
            overlap_end_time = min(record_duration, (window_end - record_start).total_seconds())
            
            # Convert to sample indices (1-indexed for adi library)
            start_sample = int(overlap_start_time / record.tick_dt) + 1
            end_sample = int(overlap_end_time / record.tick_dt) + 1
            
            # Confirm within record boundaries
            start_sample = max(1, start_sample)
            end_sample = min(record.n_ticks, end_sample)
            
            matching_records.append({
                'record_id': record.id,
                'record_start_datetime': record_start,
                'record_end_datetime': record_end,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'start_time_in_record': overlap_start_time,
                'end_time_in_record': overlap_end_time,
                'dt': record.tick_dt
            })
    
    return matching_records

def extract_comment_window(f, comment, seconds_before, seconds_after, channel_ids=None, **kwargs):
    """
    Extract data from specified channels around a comment event.
    
    Parameters
    ----------
    f : adi.File
        An opened ADI file object from adi.read_file()
    comment : dict or pd.Series
        Comment information that must contain at least:
        - 'datetime': datetime of the comment
        - 'text': comment text
    seconds_before : float
        Seconds before the comment to include in extraction
    seconds_after : float  
        Seconds after the comment to include in extraction
    channel_ids : int, list of int, or None, optional
        Channel ID(s) to extract (1-based indexing). If None, extracts all channels.
    **kwargs : dict
        Additional key-value pairs to include in the returned dictionary
    
    Returns
    -------
    dict
        Dictionary containing:
        - event: Comment text
        - time: Comment datetime
        - data: DataFrame with columns:
            - relative_time: Seconds relative to comment time (negative = before)
            - datetime: Absolute datetime for each sample
            - ch{id}_{name}_{units}: Channel data columns
        - records_used: List of record IDs that contained data (empty if no data)
        - metadata: Dictionary containing:
            - comment: Original comment object
            - s_before: seconds_before value
            - s_after: seconds_after value
            - n_samples: Number of samples extracted (0 if no data)
            - channel_ids: Channel IDs that were extracted
            - records: List of record IDs used
            - message: Warning message if no data found
        - Any additional kwargs passed to the function
    
    Notes
    -----
    This function handles extraction across record boundaries. If the time window
    spans multiple records, data is automatically concatenated. If no data exists
    in the specified window, an empty DataFrame is returned with an appropriate
    message in the metadata.
    
    Examples
    --------
    >>> f = adi.read_file('data.adicht')
    >>> comments_df = extract_comments(f)
    >>> # Extract 30 seconds before and 60 seconds after first comment
    >>> result = extract_comment_window(f, comments_df.iloc[0], 30, 60, channel_ids=[1, 2])
    >>> data = result['data']
    >>> print(f"Extracted {result['metadata']['n_samples']} samples")
    """
    # Could be a comment object or comment index
    # Could make this extract event window, instead. check whether the arg is a datetime or comment (object or index)
    
    event_time = convert_time(comment['datetime'], f)

    # Edge case - window around comment extends into other records. Data may be discontinuous.
    window = create_window(event_time, seconds_before, seconds_after)
    records_info = find_records(f, window)

    results = {
            'event': comment['text'],
            'time': event_time,
            'data': pd.DataFrame(),
            'records_used': [],
            'metadata': {
                'message': 'No data found in time window',
                's_before': seconds_before,
                's_after': seconds_after
            },
            **kwargs
        }
    
    if not records_info:
        return results
    
    data_df = _extract_data(f, records_info, event_time, channel_ids)
    
    results.update({
        'event': comment['text'],
        'time': event_time,
        'data': data_df,
        'metadata': {
            'comment': comment,
            's_before': seconds_before,
            's_after': seconds_after,
            'n_samples': len(data_df),
            'channel_ids': channel_ids,
            'records': [r['record_id'] for r in records_info]
        },
        **kwargs
    })

    return results

def extract_window(f, window, channel_ids=None, **kwargs):
    """
    Extract data within a specified time window.
    
    Parameters
    ----------
    f : adi.File
        An opened ADI file object from adi.read_file()
    window : tuple or list of 2 elements
        Time window as (start_time, end_time). Each time can be:
        - datetime object
        - float/int representing seconds since midnight
        - np.ndarray (for R/reticulate compatibility)
    channel_ids : int, list of int, or None, optional
        Channel ID(s) to extract (1-based indexing). If None, extracts all channels.
    **kwargs : dict
        Additional key-value pairs to include in the returned dictionary
    
    Returns
    -------
    dict
        Dictionary containing:
        - time: Window boundaries as list of datetime objects
        - data: DataFrame with columns:
            - relative_time: Seconds relative to window start
            - datetime: Absolute datetime for each sample
            - ch{id}_{name}_{units}: Channel data columns
        - records_used: List of record IDs (empty if no data found)
        - metadata: Dictionary containing:
            - n_samples: Number of samples extracted
            - channel_ids: Channel IDs that were extracted
            - records: List of record IDs used
            - message: 'No data found in time window' if applicable
        - Any additional kwargs passed to the function
    
    Notes
    -----
    This function handles extraction across record boundaries. Data is 
    automatically concatenated if the window spans multiple records. Times
    in the output DataFrame are relative to the start of the window.
    
    Examples
    --------
    >>> f = adi.read_file('data.adicht')
    >>> # Extract data between two specific times
    >>> start = datetime(2024, 1, 1, 12, 0, 0)
    >>> end = datetime(2024, 1, 1, 12, 5, 0)
    >>> result = extract_window(f, (start, end), channel_ids=[1, 2])
    >>> 
    >>> # Using seconds since midnight
    >>> result = extract_window(f, (3600, 3900), channel_ids=1)  # 1:00-1:05 AM
    """
    # Edge case - window around comment extends into other records. Data may be discontinuous.
    results = {
            'time': window,
            'data': pd.DataFrame(),
            'records_used': [],
            'metadata': {
                'message': 'No data found in time window',
            },
            **kwargs
        }

    window = [convert_time(time, f) for time in window]
    records_info = find_records(f, window)

    if not records_info:
        return results
    
    data_df = _extract_data(f, records_info, window[0], channel_ids)
    
    results.update({
        'time': window,
        'data': data_df,
        'metadata': {
            'n_samples': len(data_df),
            'channel_ids': channel_ids,
            'records': [r['record_id'] for r in records_info]
        }
        })

    return results

def _extract_data(f, records_info, basis_time, channel_ids=None):
    """
    Extract data from specified channels across multiple records.
    
    Parameters
    ----------
    f : adi.File
        An opened ADI file object from adi.read_file()
    records_info : list of dict
        List of record information dictionaries from find_records() containing
        record_id, start_sample, end_sample, dt, etc.
    basis_time : datetime
        Reference time for calculating relative times. All times in the output
        will be relative to this datetime.
    channel_ids : int, list of int, or None, optional
        Channel ID(s) to extract (1-based indexing). Can be:
        - None: extracts all channels from the first record
        - int: extracts a single channel
        - list of int: extracts multiple specified channels
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing extracted data with columns:
        - relative_time: Time in seconds relative to basis_time
        - datetime: Absolute datetime for each sample
        - ch{id}_{name}_{units}: One column per channel with the actual data
    
    Notes
    -----
    This is an internal function used by the public API functions. It handles:
    - Data extraction across record boundaries
    - Time alignment when data spans multiple records  
    - Missing channels (fills with NaN)
    - Channel naming with units included
    
    The function assumes channel configuration is consistent across records
    when channel_ids is None.
    """
    # Determine which channels to extract
    if channel_ids is None:
        # Get all channels from the first relevant record (assuming consistent across records)
        first_record = f.records[records_info[0]['record_id'] - 1]
        channel_ids = list(range(1, first_record.n_channels + 1))
    elif type(channel_ids) is not list:
        channel_ids = [channel_ids]
    
    # Collect all data
    all_data_frames = []
    
    for record in records_info:
        
        # Create time array for this record segment
        # +1 for inclusion
        n_samples = record['end_sample'] - record['start_sample'] + 1
        sample_times_in_record = np.arange(n_samples) * record['dt'] + record['start_time_in_record']
        
        # Convert to absolute times, then to seconds relative to start of window
        record_start = record['record_start_datetime']
        absolute_times = [record_start + timedelta(seconds=t) for t in sample_times_in_record]
        relative_times = [(t - basis_time).total_seconds() for t in absolute_times]
        
        # Create DataFrame for this record segment
        segment_data = {'relative_time': relative_times,
                    'datetime': absolute_times}
        
        # Extract data for each channel
        for channel_id in channel_ids:
            channel_id = int(channel_id)
            try:
                data = f.channels[channel_id].get_data(
                    record_id = record['record_id'], 
                    start_sample = record['start_sample'], 
                    stop_sample = record['end_sample']
                    )
                
                # Get channel name if available
                channel = f.channels[channel_id - 1]
                channel_name = f"ch{channel_id}_{channel.name}_{channel.units[record['record_id']-1]}" if channel.name else f"ch{channel_id}"
                
                segment_data[channel_name] = data
            except Exception as e:
                # Handle case where channel might not exist in this record
                segment_data[f"ch{channel_id}"] = [np.nan] * n_samples
        
        all_data_frames.append(pd.DataFrame(segment_data))
    
    # Concatenate all segments
    if all_data_frames:
        data_df = pd.concat(all_data_frames, ignore_index=True)
        # Sort by time to ensure proper ordering
        data_df = data_df.sort_values('relative_time').reset_index(drop=True)
    else:
        data_df = pd.DataFrame()

    return data_df