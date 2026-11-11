def urljoin_(base, url):
    """
    Reproduce urllib.parse.urljoin from scratch, implementing RFC 3986 reference resolution.
    """
    if not base:
        return url

    # Parse the base URL into components
    base_scheme, base_netloc, base_path, base_query, base_fragment = _parse_url(base)
    
    # Parse the reference URL into components
    ref_scheme, ref_netloc, ref_path, ref_query, ref_fragment = _parse_url(url)

    # If the reference has a scheme, it's an absolute URL
    if ref_scheme:
        return _build_url(ref_scheme, ref_netloc, ref_path, ref_query, ref_fragment)

    # If the reference has a netloc (starts with //), it's scheme-relative
    if ref_netloc:
        return _build_url(base_scheme, ref_netloc, ref_path, ref_query, ref_fragment)

    # If the reference path is empty and query is empty, it's a fragment-only reference
    if not ref_path and not ref_query:
        # Fragment-only: return base with new fragment
        return _build_url(base_scheme, base_netloc, base_path, base_query, ref_fragment)

    # If the reference path is empty but query is present, it's a query-only reference
    if not ref_path:
        # Query-only: return base with new query and fragment
        return _build_url(base_scheme, base_netloc, base_path, ref_query, ref_fragment)

    # If the reference path starts with '/', it's an absolute path
    if ref_path.startswith('/'):
        return _build_url(base_scheme, base_netloc, ref_path, ref_query, ref_fragment)

    # Otherwise, it's a relative path
    # Merge the base path with the reference path
    merged_path = _merge_paths(base_path, ref_path)
    # Normalize the merged path
    normalized_path = _normalize_path(merged_path)
    return _build_url(base_scheme, base_netloc, normalized_path, ref_query, ref_fragment)


def _parse_url(url):
    """
    Parse a URL into (scheme, netloc, path, query, fragment).
    """
    scheme = ''
    netloc = ''
    path = ''
    query = ''
    fragment = ''

    # Remove fragment
    if '#' in url:
        url, fragment = url.split('#', 1)

    # Remove query
    if '?' in url:
        url, query = url.split('?', 1)

    # Check for scheme
    if ':' in url:
        # Find the first colon that is part of the scheme
        # Scheme must be at the beginning and contain only valid scheme characters
        i = 0
        while i < len(url) and (url[i].isalnum() or url[i] in '+-.'):
            i += 1
        if i > 0 and i < len(url) and url[i] == ':':
            scheme = url[:i]
            url = url[i+1:]

    # Check for netloc
    if url.startswith('//'):
        url = url[2:]
        # Netloc ends at the first '/', '?', or '#'
        i = 0
        while i < len(url) and url[i] not in '/?#':
            i += 1
        netloc = url[:i]
        url = url[i:]

    # The rest is the path
    path = url

    return scheme, netloc, path, query, fragment


def _build_url(scheme, netloc, path, query, fragment):
    """
    Build a URL from components.
    """
    result = ''
    if scheme:
        result += scheme + ':'
    if netloc:
        result += '//' + netloc
    result += path
    if query:
        result += '?' + query
    if fragment:
        result += '#' + fragment
    return result


def _merge_paths(base_path, ref_path):
    """
    Merge the base path with the reference path for relative references.
    """
    if not base_path:
        return ref_path

    # If base path ends with '/', just append
    if base_path.endswith('/'):
        return base_path + ref_path

    # Otherwise, remove the last segment of base path and append ref path
    # Find the last '/'
    last_slash = base_path.rfind('/')
    if last_slash == -1:
        # No slash in base path, so the entire base path is the last segment
        return ref_path
    else:
        return base_path[:last_slash + 1] + ref_path


def _normalize_path(path):
    """
    Normalize a path by resolving '.' and '..' segments.
    """
    if not path:
        return path

    # Split the path into segments
    segments = path.split('/')
    result = []

    for segment in segments:
        if segment == '.':
            # Skip '.' segments
            continue
        elif segment == '..':
            # Remove the last segment if possible
            if result and result[-1] != '':
                result.pop()
            # If the last segment is empty (which happens with leading '/'), we can't go up
            # But we should still handle it properly
        else:
            result.append(segment)

    # Rejoin the segments
    normalized = '/'.join(result)

    # Preserve leading slash if the original path had one
    if path.startswith('/') and not normalized.startswith('/'):
        normalized = '/' + normalized

    # Handle the case where the path ends with '/'
    # If the original path ended with '/' and the last segment was '.' or '..', we need to add '/'
    if path.endswith('/') and not normalized.endswith('/'):
        # Check if the last segment was '.' or '..'
        last_segment = path.rstrip('/').split('/')[-1] if path.rstrip('/') else ''
        if last_segment in ('.', '..'):
            normalized += '/'

    return normalized
