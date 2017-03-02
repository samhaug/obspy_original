# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Stream objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import copy
import fnmatch
import math
import os
import pickle
import warnings
from glob import glob, has_magic

from pkg_resources import load_entry_point
import numpy as np

from obspy.core import compatibility
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.base import (ENTRY_POINTS, _get_function_from_entry_point,
                                  _read_from_plugin, download_to_file)
from obspy.core.util.decorator import (deprecated, map_example_filename,
                                       raise_if_masked, uncompress_file)
from obspy.core.util.misc import get_window_times


_headonly_warning_msg = (
    "Keyword headonly cannot be combined with starttime, endtime or dtype.")


@map_example_filename("pathname_or_url")
def read(pathname_or_url=None, format=None, headonly=False, starttime=None,
         endtime=None, nearest_sample=True, dtype=None, apply_calib=False,
         **kwargs):
    """
    Read waveform files into an ObsPy Stream object.
    """
    # add default parameters to kwargs so sub-modules may handle them
    kwargs['starttime'] = starttime
    kwargs['endtime'] = endtime
    kwargs['nearest_sample'] = nearest_sample
    # create stream
    st = Stream()
    if pathname_or_url is None:
        # if no pathname or URL specified, return example stream
        st = _create_example_stream(headonly=headonly)
    elif not isinstance(pathname_or_url, (str, native_str)):
        # not a string - we assume a file-like object
        pathname_or_url.seek(0)
        try:
            # first try reading directly
            stream = _read(pathname_or_url, format, headonly, **kwargs)
            st.extend(stream.traces)
        except TypeError:
            # if this fails, create a temporary file which is read directly
            # from the file system
            pathname_or_url.seek(0)
            with NamedTemporaryFile() as fh:
                fh.write(pathname_or_url.read())
                st.extend(_read(fh.name, format, headonly, **kwargs).traces)
        pathname_or_url.seek(0)
    elif "://" in pathname_or_url:
        # some URL
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        with NamedTemporaryFile(suffix=suffix) as fh:
            download_to_file(url=pathname_or_url, filename_or_buffer=fh)
            st.extend(_read(fh.name, format, headonly, **kwargs).traces)
    else:
        # some file name
        pathname = pathname_or_url
        for file in sorted(glob(pathname)):
            st.extend(_read(file, format, headonly, **kwargs).traces)
        if len(st) == 0:
            # try to give more specific information why the stream is empty
            if has_magic(pathname) and not glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)
            # Only raise error if no start/end time has been set. This
            # will return an empty stream if the user chose a time window with
            # no data in it.
            # XXX: Might cause problems if the data is faulty and the user
            # set start/end time. Not sure what to do in this case.
            elif not starttime and not endtime:
                raise Exception("Cannot open file/files: %s" % pathname)
    # Trim if times are given.
    if headonly and (starttime or endtime or dtype):
        warnings.warn(_headonly_warning_msg, UserWarning)
        return st
    if starttime:
        st._ltrim(starttime, nearest_sample=nearest_sample)
    if endtime:
        st._rtrim(endtime, nearest_sample=nearest_sample)
    # convert to dtype if given
    if dtype:
        # For compatibility with NumPy 1.4
        if isinstance(dtype, str):
            dtype = native_str(dtype)
        for tr in st:
            tr.data = np.require(tr.data, dtype)
    # applies calibration factor
    if apply_calib:
        for tr in st:
            tr.data = tr.data * tr.stats.calib
    return st


@uncompress_file
def _read(filename, format=None, headonly=False, **kwargs):
    """
    Read a single file into a ObsPy Stream object.
    """
    stream, format = _read_from_plugin('waveform', filename, format=format,
                                       headonly=headonly, **kwargs)
    # set _format identifier for each element
    for trace in stream:
        trace.stats._format = format
    return stream


def _create_example_stream(headonly=False):
    """
    Create an example stream.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not headonly:
        path = os.path.join(data_dir, "example.npz")
        data = np.load(path)
    st = Stream()
    for channel in ["EHZ", "EHN", "EHE"]:
        header = {'network': "BW",
                  'station': "RJOB",
                  'location': "",
                  'npts': 3000,
                  'starttime': UTCDateTime(2009, 8, 24, 0, 20, 3),
                  'sampling_rate': 100.0,
                  'calib': 1.0,
                  'back_azimuth': 100.0,
                  'inclination': 30.0}
        header['channel'] = channel
        if not headonly:
            st.append(Trace(data=data[channel], header=header))
        else:
            st.append(Trace(header=header))
    from obspy import read_inventory
    inv = read_inventory(os.path.join(data_dir, "BW_RJOB.xml"))
    st.attach_response(inv)
    return st


class Stream(object):
    """
    List like object of multiple ObsPy Trace objects.

    """

    def __init__(self, traces=None):
        self.traces = []
        if isinstance(traces, Trace):
            traces = [traces]
        if traces:
            self.traces.extend(traces)

    def __add__(self, other):
        """
        """
        if isinstance(other, Trace):
            other = Stream([other])
        if not isinstance(other, Stream):
            raise TypeError
        traces = self.traces + other.traces
        return self.__class__(traces=traces)

    def __iadd__(self, other):
        """
        """
        if isinstance(other, Trace):
            other = Stream([other])
        if not isinstance(other, Stream):
            raise TypeError
        self.extend(other.traces)
        return self

    def __mul__(self, num):
        """
        """
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        from obspy import Stream
        st = Stream()
        for _i in range(num):
            st += self.copy()
        return st

    def __iter__(self):
        """
        """
        return list(self.traces).__iter__()

    def __nonzero__(self):
        """
        A Stream is considered zero if has no Traces.
        """
        return bool(len(self.traces))

    def __len__(self):
        """
        """
        return len(self.traces)

    count = __len__

    def __str__(self, extended=False):
        """
        """
        # get longest id
        if self.traces:
            id_length = self and max(len(tr.id) for tr in self) or 0
        else:
            id_length = 0
        out = str(len(self.traces)) + ' Trace(s) in Stream:\n'
        if len(self.traces) <= 20 or extended is True:
            out = out + "\n".join([_i.__str__(id_length) for _i in self])
        else:
            out = out + "\n" + self.traces[0].__str__() + "\n" + \
                '...\n(%i other traces)\n...\n' % (len(self.traces) - 2) + \
                self.traces[-1].__str__() + '\n\n[Use "print(' + \
                'Stream.__str__(extended=True))" to print all Traces]'
        return out

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__(extended=p.verbose))

    def __eq__(self, other):
        """
        """
        if not isinstance(other, Stream):
            return False

        # this is maybe still not 100% satisfactory, the question here is if
        # two streams should be the same in comparison if one of the streams
        # has a duplicate trace. Using sets at the moment, two equal traces
        # in one of the Streams would lead to two non-equal Streams.
        # This is a bit more conservative and most likely the expected behavior
        # in most cases.
        self_sorted = self.select()
        self_sorted.sort()
        other_sorted = other.select()
        other_sorted.sort()
        if self_sorted.traces != other_sorted.traces:
            return False

        return True

    def __ne__(self, other):
        """
        """
        # Calls __eq__() and returns the opposite.
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    def __le__(self, other):
        """
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    def __gt__(self, other):
        """
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    def __ge__(self, other):
        """
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    def __setitem__(self, index, trace):
        """
        """
        self.traces.__setitem__(index, trace)

    def __getitem__(self, index):
        """
        """
        if isinstance(index, slice):
            return self.__class__(traces=self.traces.__getitem__(index))
        else:
            return self.traces.__getitem__(index)

    def __delitem__(self, index):
        """
        """
        return self.traces.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        """
        """
        # see also https://docs.python.org/3/reference/datamodel.html
        return self.__class__(traces=self.traces[max(0, i):max(0, j):k])

    def append(self, trace):
        """
        """
        if isinstance(trace, Trace):
            self.traces.append(trace)
        else:
            msg = 'Append only supports a single Trace object as an argument.'
            raise TypeError(msg)
        return self

    def extend(self, trace_list):
        """
        """
        if isinstance(trace_list, list):
            for _i in trace_list:
                # Make sure each item in the list is a trace.
                if not isinstance(_i, Trace):
                    msg = 'Extend only accepts a list of Trace objects.'
                    raise TypeError(msg)
            self.traces.extend(trace_list)
        elif isinstance(trace_list, Stream):
            self.traces.extend(trace_list.traces)
        else:
            msg = 'Extend only supports a list of Trace objects as argument.'
            raise TypeError(msg)
        return self

    @deprecated(
        "'getGaps' has been renamed to "  # noqa
        "'get_gaps'. Use that instead.")
    def getGaps(self, *args, **kwargs):
        '''
        DEPRECATED: 'getGaps' has been renamed to
        'get_gaps'. Use that instead.
        '''
        return self.get_gaps(*args, **kwargs)

    def get_gaps(self, min_gap=None, max_gap=None):
        """
        """
        # Create shallow copy of the traces to be able to sort them later on.
        copied_traces = copy.copy(self.traces)
        self.sort()
        gap_list = []
        for _i in range(len(self.traces) - 1):
            # skip traces with different network, station, location or channel
            if self.traces[_i].id != self.traces[_i + 1].id:
                continue
            # different sampling rates should always result in a gap or overlap
            if self.traces[_i].stats.delta == self.traces[_i + 1].stats.delta:
                same_sampling_rate = True
            else:
                same_sampling_rate = False
            stats = self.traces[_i].stats
            stime = stats['endtime']
            etime = self.traces[_i + 1].stats['starttime']
            # last sample of earlier trace represents data up to time of last
            # sample (stats.endtime) plus one delta
            delta = etime.timestamp - (stime.timestamp + stats.delta)
            # Check that any overlap is not larger than the trace coverage
            if delta < 0:
                temp = self.traces[_i + 1].stats['endtime'].timestamp - \
                    etime.timestamp
                if (delta * -1) > temp:
                    delta = -1 * temp
            # Check gap/overlap criteria
            if min_gap and delta < min_gap:
                continue
            if max_gap and delta > max_gap:
                continue
            # Number of missing samples
            nsamples = int(compatibility.round_away(math.fabs(delta) *
                                                    stats['sampling_rate']))
            if delta < 0:
                nsamples = -nsamples
            # skip if is equal to delta (1 / sampling rate)
            if same_sampling_rate and nsamples == 0:
                continue
            gap_list.append([stats['network'], stats['station'],
                             stats['location'], stats['channel'],
                             stime, etime, delta, nsamples])
        # Set the original traces to not alter the stream object.
        self.traces = copied_traces
        return gap_list

    def insert(self, position, object):
        """
        Insert either a single Trace or a list of Traces before index.

        """
        if isinstance(object, Trace):
            self.traces.insert(position, object)
        elif isinstance(object, list):
            # Make sure each item in the list is a trace.
            for _i in object:
                if not isinstance(_i, Trace):
                    msg = 'Trace object or a list of Trace objects expected!'
                    raise TypeError(msg)
            # Insert each item of the list.
            for _i in range(len(object)):
                self.traces.insert(position + _i, object[_i])
        elif isinstance(object, Stream):
            self.insert(position, object.traces)
        else:
            msg = 'Only accepts a Trace object or a list of Trace objects.'
            raise TypeError(msg)
        return self

    def plot(self, *args, **kwargs):
        """
        """

    def pop(self, index=(-1)):
        """
        Remove and return the Trace object specified by index from the Stream.

        If no index is given, remove the last Trace. Passes on the pop() to
        self.traces.

        :param index: Index of the Trace object to be returned and removed.
        :returns: Removed Trace.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> tr = st.pop()
        >>> print(st)  # doctest: +ELLIPSIS
        2 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        """
        return self.traces.pop(index)

    @deprecated(
        "'printGaps' has been renamed to "  # noqa
        "'print_gaps'. Use that instead.")
    def printGaps(self, *args, **kwargs):
        '''
        DEPRECATED: 'printGaps' has been renamed to
        'print_gaps'. Use that instead.
        '''
        return self.print_gaps(*args, **kwargs)

    def print_gaps(self, min_gap=None, max_gap=None):
        """
        """
        result = self.get_gaps(min_gap, max_gap)
        print("%-17s %-27s %-27s %-15s %-8s" % ('Source', 'Last Sample',
                                                'Next Sample', 'Delta',
                                                'Samples'))
        gaps = 0
        overlaps = 0
        for r in result:
            if r[6] > 0:
                gaps += 1
            else:
                overlaps += 1
            print("%-17s %-27s %-27s %-15.6f %-8d" % ('.'.join(r[0:4]),
                                                      r[4], r[5], r[6], r[7]))
        print("Total: %d gap(s) and %d overlap(s)" % (gaps, overlaps))

    def remove(self, trace):
        """
        """
        self.traces.remove(trace)
        return self

    def reverse(self):
        """
        """
        self.traces.reverse()
        return self

    def sort(self, keys=['network', 'station', 'location', 'channel',
                         'starttime', 'endtime'], reverse=False):
        """
        """
        # check if list
        msg = "keys must be a list of strings. Always available items to " + \
            "sort after: \n'network', 'station', 'channel', 'location', " + \
            "'starttime', 'endtime', 'sampling_rate', 'npts', 'dataquality'"
        if not isinstance(keys, list):
            raise TypeError(msg)
        # Loop over all keys in reversed order.
        for _i in keys[::-1]:
            self.traces.sort(key=lambda x: x.stats[_i], reverse=reverse)
        return self

    def write(self, filename, format, **kwargs):
        """
        """
        # Check all traces for masked arrays and raise exception.
        for trace in self.traces:
            if isinstance(trace.data, np.ma.masked_array):
                msg = 'Masked array writing is not supported. You can use ' + \
                      'np.array.filled() to convert the masked array to a ' + \
                      'normal array.'
                raise NotImplementedError(msg)
        format = format.upper()
        try:
            # get format specific entry point
            format_ep = ENTRY_POINTS['waveform_write'][format]
            # search writeFormat method for given entry point
            write_format = load_entry_point(
                format_ep.dist.key,
                'obspy.plugin.waveform.%s' % (format_ep.name), 'writeFormat')
        except (IndexError, ImportError, KeyError):
            msg = "Writing format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format,
                                   ', '.join(ENTRY_POINTS['waveform_write'])))
        write_format(self, filename, **kwargs)

    def trim(self, starttime=None, endtime=None, pad=False,
             nearest_sample=True, fill_value=None):
        """
        Cut all traces of this Stream object to given start and end time.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Specify the start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Specify the end time.
        :type pad: bool, optional
        :param pad: Gives the possibility to trim at time points outside the
            time frame of the original trace, filling the trace with the
            given ``fill_value``. Defaults to ``False``.
        :type nearest_sample: bool, optional
        :param nearest_sample: If set to ``True``, the closest sample is
            selected, if set to ``False``, the outer (previous sample for a
            start time border, next sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

            Given the following trace containing 4 samples, "|" are the
            sample points, "A" is the requested starttime::

                |        A|         |         |

            ``nearest_sample=True`` will select the second sample point,
            ``nearest_sample=False`` will select the first sample point.

        :type fill_value: int, float or ``None``, optional
        :param fill_value: Fill value for gaps. Defaults to ``None``. Traces
            will be converted to NumPy masked arrays if no value is given and
            gaps are present.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data will no longer be accessible afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.

        .. rubric:: Example

        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> dt = UTCDateTime("2009-08-24T00:20:20")
        >>> st.trim(dt, dt + 5)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        BW.RJOB..EHN | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        BW.RJOB..EHE | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        """
        if not self:
            return
        # select start/end time fitting to a sample point of the first trace
        if nearest_sample:
            tr = self.traces[0]
            if starttime:
                delta = compatibility.round_away(
                    (starttime - tr.stats.starttime) * tr.stats.sampling_rate)
                starttime = tr.stats.starttime + delta * tr.stats.delta
            if endtime:
                delta = compatibility.round_away(
                    (endtime - tr.stats.endtime) * tr.stats.sampling_rate)
                # delta is negative!
                endtime = tr.stats.endtime + delta * tr.stats.delta
        for trace in self.traces:
            trace.trim(starttime, endtime, pad=pad,
                       nearest_sample=nearest_sample, fill_value=fill_value)
        # remove empty traces after trimming
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        return self

    def _ltrim(self, starttime, pad=False, nearest_sample=True):
        """
        """
        for trace in self.traces:
            trace.trim(starttime=starttime, pad=pad,
                       nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def _rtrim(self, endtime, pad=False, nearest_sample=True):
        """
        """
        for trace in self.traces:
            trace.trim(endtime=endtime, pad=pad, nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def cutout(self, starttime, endtime):
        """
        Cut the given time range out of all traces of this Stream object.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of time span to remove from stream.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of time span to remove from stream.

        .. rubric:: Example

        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> t1 = UTCDateTime("2009-08-24T00:20:06")
        >>> t2 = UTCDateTime("2009-08-24T00:20:11")
        >>> st.cutout(t1, t2)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        6 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 301 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 301 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 301 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:11.000000Z ... | 100.0 Hz, 2200 samples
        BW.RJOB..EHN | 2009-08-24T00:20:11.000000Z ... | 100.0 Hz, 2200 samples
        BW.RJOB..EHE | 2009-08-24T00:20:11.000000Z ... | 100.0 Hz, 2200 samples
        """
        tmp = self.slice(endtime=starttime, keep_empty_traces=False)
        tmp += self.slice(starttime=endtime, keep_empty_traces=False)
        self.traces = tmp.traces
        return self

    def slice(self, starttime=None, endtime=None, keep_empty_traces=False,
              nearest_sample=True):
        """
        """
        tmp = copy.copy(self)
        tmp.traces = []
        new = tmp.copy()
        for trace in self:
            sliced_trace = trace.slice(starttime=starttime, endtime=endtime,
                                       nearest_sample=nearest_sample)
            if keep_empty_traces is False and not sliced_trace.stats.npts:
                continue
            new.append(sliced_trace)
        return new

    def slide(self, window_length, step, offset=0,
              include_partial_windows=False, nearest_sample=True):
        """
        Generator yielding equal length sliding windows of the Stream.

        Please keep in mind that it only returns a new view of the original
        data. Any modifications are applied to the original data as well. If
        you don't want this you have to create a copy of the yielded
        windows. Also be aware that if you modify the original data and you
        have overlapping windows, all following windows are affected as well.

        Not all yielded windows must have the same number of traces. The
        algorithm will determine the maximal temporal extents by analysing
        all Traces and then creates windows based on these times.

        .. rubric:: Example

        >>> import obspy
        >>> st = obspy.read()
        >>> for windowed_st in st.slide(window_length=10.0, step=10.0):
        ...     print(windowed_st)
        ...     print("---")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        3 Trace(s) in Stream:
        ... | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:13.000000Z | ...
        ... | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:13.000000Z | ...
        ... | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:13.000000Z | ...
        ---
        3 Trace(s) in Stream:
        ... | 2009-08-24T00:20:13.000000Z - 2009-08-24T00:20:23.000000Z | ...
        ... | 2009-08-24T00:20:13.000000Z - 2009-08-24T00:20:23.000000Z | ...
        ... | 2009-08-24T00:20:13.000000Z - 2009-08-24T00:20:23.000000Z | ...


        :param window_length: The length of each window in seconds.
        :type window_length: float
        :param step: The step between the start times of two successive
            windows in seconds. Can be negative if an offset is given.
        :type step: float
        :param offset: The offset of the first window in seconds relative to
            the start time of the whole interval.
        :type offset: float
        :param include_partial_windows: Determines if windows that are
            shorter then 99.9 % of the desired length are returned.
        :type include_partial_windows: bool
        :param nearest_sample: If set to ``True``, the closest sample is
            selected, if set to ``False``, the outer (previous sample for a
            start time border, next sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

            Given the following trace containing 4 samples, "|" are the
            sample points, "A" is the requested starttime::

                |        A|         |         |

            ``nearest_sample=True`` will select the second sample point,
            ``nearest_sample=False`` will select the first sample point.
        :type nearest_sample: bool, optional
        """
        starttime = min(tr.stats.starttime for tr in self)
        endtime = max(tr.stats.endtime for tr in self)
        windows = get_window_times(
            starttime=starttime,
            endtime=endtime,
            window_length=window_length,
            step=step,
            offset=offset,
            include_partial_windows=include_partial_windows)

        if len(windows) < 1:
            raise StopIteration

        for start, stop in windows:
            temp = self.slice(start, stop,
                              nearest_sample=nearest_sample)
            # It might happen that there is a time frame where there are no
            # windows, e.g. two traces separated by a large gap.
            if not temp:
                continue
            yield temp

        raise StopIteration

    def select(self, network=None, station=None, location=None, channel=None,
               sampling_rate=None, npts=None, component=None, id=None):
        """
        Return new Stream object only with these traces that match the given
        stats criteria (e.g. all traces with ``channel="EHZ"``).

        .. rubric:: Examples

        >>> from obspy import read
        >>> st = read()
        >>> st2 = st.select(station="R*")
        >>> print(st2)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples

        >>> st2 = st.select(id="BW.RJOB..EHZ")
        >>> print(st2)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples

        >>> st2 = st.select(component="Z")
        >>> print(st2)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples

        >>> st2 = st.select(network="CZ")
        >>> print(st2)  # doctest: +NORMALIZE_WHITESPACE
        0 Trace(s) in Stream:

        .. warning::
            A new Stream object is returned but the traces it contains are
            just aliases to the traces of the original stream. Does not copy
            the data but only passes a reference.

        All keyword arguments except for ``component`` are tested directly
        against the respective entry in the :class:`~obspy.core.trace.Stats`
        dictionary.

        If a string for ``component`` is given (should be a single letter) it
        is tested against the last letter of the ``Trace.stats.channel`` entry.

        Alternatively, ``channel`` may have the last one or two letters
        wildcarded (e.g. ``channel="EH*"``) to select all components with a
        common band/instrument code.

        All other selection criteria that accept strings (network, station,
        location) may also contain Unix style wildcards (``*``, ``?``, ...).
        """
        # make given component letter uppercase (if e.g. "z" is given)
        if component and channel:
            component = component.upper()
            channel = channel.upper()
            if channel[-1] != "*" and component != channel[-1]:
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise ValueError(msg)
        traces = []
        for trace in self:
            # skip trace if any given criterion is not matched
            if id and not fnmatch.fnmatch(trace.id.upper(), id.upper()):
                continue
            if network is not None:
                if not fnmatch.fnmatch(trace.stats.network.upper(),
                                       network.upper()):
                    continue
            if station is not None:
                if not fnmatch.fnmatch(trace.stats.station.upper(),
                                       station.upper()):
                    continue
            if location is not None:
                if not fnmatch.fnmatch(trace.stats.location.upper(),
                                       location.upper()):
                    continue
            if channel is not None:
                if not fnmatch.fnmatch(trace.stats.channel.upper(),
                                       channel.upper()):
                    continue
            if sampling_rate is not None:
                if float(sampling_rate) != trace.stats.sampling_rate:
                    continue
            if npts is not None and int(npts) != trace.stats.npts:
                continue
            if component is not None:
                if len(trace.stats.channel) < 3:
                    continue
                if not fnmatch.fnmatch(trace.stats.channel[-1].upper(),
                                       component.upper()):
                    continue
            traces.append(trace)
        return self.__class__(traces=traces)

    def verify(self):
        """
        Verify all traces of current Stream against available meta data.

        .. rubric:: Example

        >>> from obspy import Trace, Stream
        >>> tr = Trace(data=np.array([1, 2, 3, 4]))
        >>> tr.stats.npts = 100
        >>> st = Stream([tr])
        >>> st.verify()  #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        Exception: ntps(100) differs from data size(4)
        """
        for trace in self:
            trace.verify()
        return self

    def _merge_checks(self):
        """
        Sanity checks for merging.
        """
        sr = {}
        dtype = {}
        calib = {}
        for trace in self.traces:
            # skip empty traces
            if len(trace) == 0:
                continue
            # Check sampling rate.
            sr.setdefault(trace.id, trace.stats.sampling_rate)
            if trace.stats.sampling_rate != sr[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "sampling rates!"
                raise Exception(msg)
            # Check dtype.
            dtype.setdefault(trace.id, trace.data.dtype)
            if trace.data.dtype != dtype[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "data types!"
                raise Exception(msg)
            # Check calibration factor.
            calib.setdefault(trace.id, trace.stats.calib)
            if trace.stats.calib != calib[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "calibration factors.!"
                raise Exception(msg)

    def merge(self, method=0, fill_value=None, interpolation_samples=0,
              **kwargs):
        """
        Merge ObsPy Trace objects with same IDs.

        :type method: int, optional
        :param method: Methodology to handle overlaps/gaps of traces. Defaults
            to ``0``.
            See :meth:`obspy.core.trace.Trace.__add__` for details on
            methods ``0`` and ``1``,
            see :meth:`obspy.core.stream.Stream._cleanup` for details on
            method ``-1``. Any merge operation performs a cleanup merge as
            a first step (method ``-1``).
        :type fill_value: int, float, str or ``None``, optional
        :param fill_value: Fill value for gaps. Defaults to ``None``. Traces
            will be converted to NumPy masked arrays if no value is given and
            gaps are present. The value ``'latest'`` will use the latest value
            before the gap. If value ``'interpolate'`` is provided, missing
            values are linearly interpolated (not changing the data
            type e.g. of integer valued traces). Not used for ``method=-1``.
        :type interpolation_samples: int, optional
        :param interpolation_samples: Used only for ``method=1``. It specifies
            the number of samples which are used to interpolate between
            overlapping traces. Default to ``0``. If set to ``-1`` all
            overlapping samples are interpolated.

        Importing waveform data containing gaps or overlaps results into
        a :class:`~obspy.core.stream.Stream` object with multiple traces having
        the same identifier. This method tries to merge such traces inplace,
        thus returning nothing. Merged trace data will be converted into a
        NumPy :class:`~numpy.ma.MaskedArray` type if any gaps are present. This
        behavior may be prevented by setting the ``fill_value`` parameter.
        The ``method`` argument controls the handling of overlapping data
        values.
        """
        def listsort(order, current):
            """
            Helper method for keeping trace's ordering
            """
            try:
                return order.index(current)
            except ValueError:
                return -1

        self._cleanup(**kwargs)
        if method == -1:
            return
        # check sampling rates and dtypes
        self._merge_checks()
        # remember order of traces
        order = [id(i) for i in self.traces]
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                        'starttime', 'endtime'])
        # build up dictionary with with lists of traces with same ids
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop(0)
                # skip empty traces
                if len(trace) == 0:
                    continue
                _id = trace.get_id()
                if _id not in traces_dict:
                    traces_dict[_id] = [trace]
                else:
                    traces_dict[_id].append(trace)
        except IndexError:
            pass
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for _id in traces_dict.keys():
            cur_trace = traces_dict[_id].pop(0)
            # loop through traces of same id
            for _i in range(len(traces_dict[_id])):
                trace = traces_dict[_id].pop(0)
                # disable sanity checks because there are already done
                cur_trace = cur_trace.__add__(
                    trace, method, fill_value=fill_value, sanity_checks=False,
                    interpolation_samples=interpolation_samples)
            self.traces.append(cur_trace)

        # trying to restore order, newly created traces are placed at
        # start
        self.traces.sort(key=lambda x: listsort(order, id(x)))
        return self


    def filter(self, type, **options):
        """
        Filter the data of all traces in the Stream.

        :type type: str
        :param type: String that specifies which filter is applied (e.g.
            ``"bandpass"``). See the `Supported Filter`_ section below for
            further details.
        :param options: Necessary keyword arguments for the respective filter
            that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
            ``"bandpass"``)

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: _`Supported Filter`

        ``'bandpass'``
            Butterworth-Bandpass (uses :func:`obspy.signal.filter.bandpass`).

        ``'bandstop'``
            Butterworth-Bandstop (uses :func:`obspy.signal.filter.bandstop`).

        ``'lowpass'``
            Butterworth-Lowpass (uses :func:`obspy.signal.filter.lowpass`).

        ``'highpass'``
            Butterworth-Highpass (uses :func:`obspy.signal.filter.highpass`).

        ``'lowpass_cheby_2'``
            Cheby2-Lowpass (uses :func:`obspy.signal.filter.lowpass_cheby_2`).

        ``'lowpass_fir'`` (experimental)
            FIR-Lowpass (uses :func:`obspy.signal.filter.lowpass_fir`).

        ``'remez_fir'`` (experimental)
            Minimax optimal bandpass using Remez algorithm (uses
            :func:`obspy.signal.filter.remez_fir`).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> st.filter("highpass", freq=1.0)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            st.filter("highpass", freq=1.0)
            st.plot()
        """
        for tr in self:
            tr.filter(type, **options)
        return self

    def trigger(self, type, **options):
        """
        Run a triggering algorithm on all traces in the stream.

        :param type: String that specifies which trigger is applied (e.g.
            ``'recstalta'``). See the `Supported Trigger`_ section below for
            further details.
        :param options: Necessary keyword arguments for the respective
            trigger that will be passed on. (e.g. ``sta=3``, ``lta=10``)
            Arguments ``sta`` and ``lta`` (seconds) will be mapped to ``nsta``
            and ``nlta`` (samples) by multiplying with sampling rate of trace.
            (e.g. ``sta=3``, ``lta=10`` would call the trigger with 3 and 10
            seconds average, respectively)

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: _`Supported Trigger`

        ``'classicstalta'``
            Computes the classic STA/LTA characteristic function (uses
            :func:`obspy.signal.trigger.classic_sta_lta`).

        ``'recstalta'``
            Recursive STA/LTA
            (uses :func:`obspy.signal.trigger.recursive_sta_lta`).

        ``'recstaltapy'``
            Recursive STA/LTA written in Python (uses
            :func:`obspy.signal.trigger.recursive_sta_lta_py`).

        ``'delayedstalta'``
            Delayed STA/LTA.
            (uses :func:`obspy.signal.trigger.delayed_sta_lta`).

        ``'carlstatrig'``
            Computes the carl_sta_trig characteristic function (uses
            :func:`obspy.signal.trigger.carl_sta_trig`).

        ``'zdetect'``
            Z-detector (uses :func:`obspy.signal.trigger.z_detect`).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> st.filter("highpass", freq=1.0)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP
        >>> st.trigger('recstalta', sta=1, lta=4)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            st.filter("highpass", freq=1.0)
            st.plot()
            st.trigger('recstalta', sta=1, lta=4)
            st.plot()
        """
        for tr in self:
            tr.trigger(type, **options)
        return self

    def resample(self, sampling_rate, window='hanning', no_filter=True,
                 strict_length=False):
        """
        Resample data in all traces of stream using Fourier method.

        :type sampling_rate: float
        :param sampling_rate: The sampling rate of the resampled signal.
        :type window: array_like, callable, str, float, or tuple, optional
        :param window: Specifies the window applied to the signal in the
            Fourier domain. Defaults ``'hanning'`` window. See
            :func:`scipy.signal.resample` for details.
        :type no_filter: bool, optional
        :param no_filter: Deactivates automatic filtering if set to ``True``.
            Defaults to ``True``.
        :type strict_length: bool, optional
        :param strict_length: Leave traces unchanged for which end time of
            trace would change. Defaults to ``False``.

        .. note::

            The :class:`~Stream` object has three different methods to change
            the sampling rate of its data: :meth:`~.resample`,
            :meth:`~.decimate`, and :meth:`~.interpolate`

            Make sure to choose the most appropriate one for the problem at
            hand.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        Uses :func:`scipy.signal.resample`. Because a Fourier method is used,
        the signal is assumed to be periodic.

        .. rubric:: Example

        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st.resample(10.0)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 10.0 Hz, 300 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 10.0 Hz, 300 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 10.0 Hz, 300 samples
        """
        for tr in self:
            tr.resample(sampling_rate, window=native_str(window),
                        no_filter=no_filter, strict_length=strict_length)
        return self

    def decimate(self, factor, no_filter=False, strict_length=False):
        """
        """
        for tr in self:
            tr.decimate(factor, no_filter=no_filter,
                        strict_length=strict_length)
        return self

    def max(self):
        """
        """
        return [tr.max() for tr in self]

    def differentiate(self, method='gradient'):
        """
        """
        for tr in self:
            tr.differentiate(method=method)
        return self

    def integrate(self, method='cumtrapz', **options):
        """
        """
        for tr in self:
            tr.integrate(method=method, **options)
        return self

    @raise_if_masked
    def detrend(self, type='simple'):
        """
        """
        for tr in self:
            tr.detrend(type=type)
        return self

    def taper(self, *args, **kwargs):
        """
        """
        for tr in self:
            tr.taper(*args, **kwargs)
        return self

    def interpolate(self, *args, **kwargs):
        """
        """
        for tr in self:
            tr.interpolate(*args, **kwargs)
        return self

    def std(self):
        """
        """
        return [tr.std() for tr in self]

    def normalize(self, global_max=False):
        """
        """
        # use the same value for normalization on all traces?
        if global_max:
            norm = max([abs(value) for value in self.max()])
        else:
            norm = None
        # normalize all traces
        for tr in self:
            tr.normalize(norm=norm)
        return self

    def rotate(self, method, back_azimuth=None, inclination=None):
        """
        Rotate stream objects.

        :type method: str
        :param method: Determines the rotation method.

            ``'NE->RT'``: Rotates the North- and East-components of a
                seismogram to radial and transverse components.
            ``'RT->NE'``: Rotates the radial and transverse components of a
                seismogram to North- and East-components.
            ``'ZNE->LQT'``: Rotates from left-handed Z, North, and  East system
                to LQT, e.g. right-handed ray coordinate system.
            ``'LQT->ZNE'``: Rotates from LQT, e.g. right-handed ray coordinate
                system to left handed Z, North, and East system.

        :type back_azimuth: float, optional
        :param back_azimuth: Depends on the chosen method.
            A single float, the back azimuth from station to source in degrees.
            If not given, ``stats.back_azimuth`` will be used. It will also be
            written after the rotation is done.
        :type inclination: float, optional
        :param inclination: Inclination of the ray at the station in degrees.
            Only necessary for three component rotations. If not given,
            ``stats.inclination`` will be used. It will also be written after
            the rotation is done.
        """
        if method == "NE->RT":
            func = "rotate_ne_rt"
        elif method == "RT->NE":
            func = "rotate_rt_ne"
        elif method == "ZNE->LQT":
            func = "rotate_zne_lqt"
        elif method == "LQT->ZNE":
            func = "rotate_lqt_zne"
        else:
            raise ValueError("Method has to be one of ('NE->RT', 'RT->NE', "
                             "'ZNE->LQT', or 'LQT->ZNE').")
        # Retrieve function call from entry points
        func = _get_function_from_entry_point("rotate", func)
        # Split to get the components. No need for further checks for the
        # method as invalid methods will be caught by previous conditional.
        input_components, output_components = method.split("->")
        # Figure out inclination and back-azimuth.
        if back_azimuth is None:
            try:
                back_azimuth = self[0].stats.back_azimuth
            except:
                msg = "No back-azimuth specified."
                raise TypeError(msg)
        if len(input_components) == 3 and inclination is None:
            try:
                inclination = self[0].stats.inclination
            except:
                msg = "No inclination specified."
                raise TypeError(msg)
        # Do one of the two-component rotations.
        if len(input_components) == 2:
            input_1 = self.select(component=input_components[0])
            input_2 = self.select(component=input_components[1])
            for i_1, i_2 in zip(input_1, input_2):
                dt = 0.5 * i_1.stats.delta
                if (len(i_1) != len(i_2)) or \
                        (abs(i_1.stats.starttime - i_2.stats.starttime) > dt) \
                        or (i_1.stats.sampling_rate !=
                            i_2.stats.sampling_rate):
                    msg = "All components need to have the same time span."
                    raise ValueError(msg)
            for i_1, i_2 in zip(input_1, input_2):
                output_1, output_2 = func(i_1.data, i_2.data, back_azimuth)
                i_1.data = output_1
                i_2.data = output_2
                # Rename the components.
                i_1.stats.channel = i_1.stats.channel[:-1] + \
                    output_components[0]
                i_2.stats.channel = i_2.stats.channel[:-1] + \
                    output_components[1]
                # Add the azimuth and inclination to the stats object.
                for comp in (i_1, i_2):
                    comp.stats.back_azimuth = back_azimuth
        # Do one of the three-component rotations.
        else:
            input_1 = self.select(component=input_components[0])
            input_2 = self.select(component=input_components[1])
            input_3 = self.select(component=input_components[2])
            for i_1, i_2, i_3 in zip(input_1, input_2, input_3):
                dt = 0.5 * i_1.stats.delta
                if (len(i_1) != len(i_2)) or (len(i_1) != len(i_3)) or \
                        (abs(i_1.stats.starttime -
                             i_2.stats.starttime) > dt) or \
                        (abs(i_1.stats.starttime -
                             i_3.stats.starttime) > dt) or \
                        (i_1.stats.sampling_rate !=
                            i_2.stats.sampling_rate) or \
                        (i_1.stats.sampling_rate != i_3.stats.sampling_rate):
                    msg = "All components need to have the same time span."
                    raise ValueError(msg)
            for i_1, i_2, i_3 in zip(input_1, input_2, input_3):
                output_1, output_2, output_3 = func(
                    i_1.data, i_2.data, i_3.data, back_azimuth, inclination)
                i_1.data = output_1
                i_2.data = output_2
                i_3.data = output_3
                # Rename the components.
                i_1.stats.channel = i_1.stats.channel[:-1] + \
                    output_components[0]
                i_2.stats.channel = i_2.stats.channel[:-1] + \
                    output_components[1]
                i_3.stats.channel = i_3.stats.channel[:-1] + \
                    output_components[2]
                # Add the azimuth and inclination to the stats object.
                for comp in (i_1, i_2, i_3):
                    comp.stats.back_azimuth = back_azimuth
                    comp.stats.inclination = inclination
        return self

    def copy(self):
        """
        """
        return copy.deepcopy(self)

    def clear(self):
        """
        """
        self.traces = []
        return self

    def _cleanup(self, misalignment_threshold=1e-2):
        """
        """
        # first of all throw away all empty traces
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        # check sampling rates and dtypes
        try:
            self._merge_checks()
        except Exception as e:
            if "Can't merge traces with same ids but" in str(e):
                msg = "Incompatible traces (sampling_rate, dtype, ...) " + \
                      "with same id detected. Doing nothing."
                warnings.warn(msg)
                return
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                        'starttime', 'endtime'])
        # build up dictionary with lists of traces with same ids
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop(0)
                # add trace to respective list or create that list
                traces_dict.setdefault(trace.id, []).append(trace)
        except IndexError:
            pass
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for id_ in traces_dict.keys():
            trace_list = traces_dict[id_]
            cur_trace = trace_list.pop(0)
            delta = cur_trace.stats.delta
            allowed_micro_shift = misalignment_threshold * delta
            # work through all traces of same id
            while trace_list:
                trace = trace_list.pop(0)
                # `gap` is the deviation (in seconds) of the actual start
                # time of the second trace from the expected start time
                # (for the ideal case of directly adjacent and perfectly
                # aligned traces).
                gap = trace.stats.starttime - (cur_trace.stats.endtime + delta)
                # if `gap` is larger than the designated allowed shift,
                # we treat it as a real gap and leave as is.
                if misalignment_threshold > 0 and gap <= allowed_micro_shift:
                    # `gap` is smaller than allowed shift (or equal),
                    #  the traces could be
                    #  - overlapping without being misaligned or..
                    #  - overlapping with misalignment or..
                    #  - misaligned with a micro gap
                    # check if the sampling points are misaligned:
                    misalignment = gap % delta
                    if misalignment != 0:
                        # determine the position of the second trace's
                        # sampling points in the interval between two
                        # sampling points of first trace.
                        # a `misalign_percentage` of close to 0.0 means a
                        # sampling point of the first trace is just a bit
                        # to the left of our sampling point:
                        #
                        #  Trace 1: --|---------|---------|---------|--
                        #  Trace 2: ---|---------|---------|---------|-
                        # misalign_percentage:  0.........1
                        #
                        # a `misalign_percentage` of close to 1.0 means a
                        # sampling point of the first trace is just a bit
                        # to the right of our sampling point:
                        #
                        #  Trace 1: --|---------|---------|---------|--
                        #  Trace 2: -|---------|---------|---------|---
                        # misalign_percentage:  0.........1
                        misalign_percentage = misalignment / delta
                        if (misalign_percentage <= misalignment_threshold or
                                misalign_percentage >=
                                1 - misalignment_threshold):
                            # now we align the sampling points of both traces
                            trace.stats.starttime = (
                                cur_trace.stats.starttime +
                                round((trace.stats.starttime -
                                       cur_trace.stats.starttime) / delta) *
                                delta)
                # we have some common parts: check if consistent
                # (but only if sampling points are matching to specified
                #  accuracy, which is checked and conditionally corrected in
                #  previous code block)
                subsample_shift_percentage = (
                    trace.stats.starttime.timestamp -
                    cur_trace.stats.starttime.timestamp) % delta / delta
                subsample_shift_percentage = min(
                    subsample_shift_percentage, 1 - subsample_shift_percentage)
                if (trace.stats.starttime <= cur_trace.stats.endtime and
                        subsample_shift_percentage < misalignment_threshold):
                    # check if common time slice [t1 --> t2] is equal:
                    t1 = trace.stats.starttime
                    t2 = min(cur_trace.stats.endtime, trace.stats.endtime)
                    # if consistent: add them together
                    if np.array_equal(cur_trace.slice(t1, t2).data,
                                      trace.slice(t1, t2).data):
                        cur_trace += trace
                    # if not consistent: leave them alone
                    else:
                        self.traces.append(cur_trace)
                        cur_trace = trace
                # traces are perfectly adjacent: add them together
                elif trace.stats.starttime == cur_trace.stats.endtime + \
                        cur_trace.stats.delta:
                    cur_trace += trace
                # no common parts (gap):
                # leave traces alone and add current to list
                else:
                    self.traces.append(cur_trace)
                    cur_trace = trace
            self.traces.append(cur_trace)
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def split(self):
        """
        Split any trace containing gaps into contiguous unmasked traces.

        :rtype: :class:`obspy.core.stream.Stream`
        :returns: Returns a new stream object containing only contiguous
            unmasked.
        """
        new_stream = Stream()
        for trace in self.traces:
            new_stream.extend(trace.split())
        return new_stream


@deprecated("Renamed to '_is_pickle'. Use that instead.")
def isPickle(*args, **kwargs):  # noqa
    return _is_pickle(*args, **kwargs)


@deprecated("Renamed to '_read_pickle'. Use that instead.")
def readPickle(*args, **kwargs):  # noqa
    return _read_pickle(*args, **kwargs)


@deprecated("Renamed to '_write_pickle'. Use that instead.")
def writePickle(*args, **kwargs):  # noqa
    return _write_pickle(*args, **kwargs)


def _is_pickle(filename):  # @UnusedVariable
    """
    Check whether a file is a pickled ObsPy Stream file.

    :type filename: str
    :param filename: Name of the pickled ObsPy Stream file to be checked.
    :rtype: bool
    :return: ``True`` if pickled file.

    .. rubric:: Example

    >>> _is_pickle('/path/to/pickle.file')  # doctest: +SKIP
    True
    """
    if isinstance(filename, (str, native_str)):
        try:
            with open(filename, 'rb') as fp:
                st = pickle.load(fp)
        except:
            return False
    else:
        try:
            st = pickle.load(filename)
        except:
            return False
    return isinstance(st, Stream)


def _read_pickle(filename, **kwargs):  # @UnusedVariable
    """
    Read and return Stream from pickled ObsPy Stream file.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Name of the pickled ObsPy Stream file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.
    """
    if isinstance(filename, (str, native_str)):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
    else:
        return pickle.load(filename)


def _write_pickle(stream, filename, protocol=2, **kwargs):  # @UnusedVariable
    """
    Write a Python pickle of current stream.

    .. note::
        Writing into PICKLE format allows to store additional attributes
        appended to the current Stream object or any contained Trace.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type protocol: int, optional
    :param protocol: Pickle protocol, defaults to ``2``.
    """
    if isinstance(filename, (str, native_str)):
        with open(filename, 'wb') as fp:
            pickle.dump(stream, fp, protocol=protocol)
    else:
        pickle.dump(stream, filename, protocol=protocol)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
