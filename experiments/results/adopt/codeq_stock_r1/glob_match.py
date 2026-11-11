def glob_match(pattern, name):
    """
    Implement fnmatch.fnmatchcase semantics from scratch.
    
    Supports:
    - * : matches any run of characters (including empty)
    - ? : matches exactly one character
    - [seq] : matches any character in seq
    - [!seq] : matches any character NOT in seq
    - ] immediately after [ or [! is a literal ]
    - Unterminated [ is a literal [
    
    Matching is anchored to the whole string.
    """
    def match_here(pattern, name):
        """
        Match pattern against name, both starting at their current positions.
        Returns True if the entire remaining pattern matches the entire remaining name.
        """
        while pattern:
            c = pattern[0]
            if c == '*':
                # Skip consecutive stars
                while pattern and pattern[0] == '*':
                    pattern = pattern[1:]
                if not pattern:
                    return True
                # Try matching the rest of the pattern against every possible
                # suffix of name (including empty)
                for i in range(len(name) + 1):
                    if match_here(pattern, name[i:]):
                        return True
                return False
            elif c == '?':
                if not name:
                    return False
                pattern = pattern[1:]
                name = name[1:]
            elif c == '[':
                # Parse the character class
                # Check for unterminated [
                # Find the closing ]
                # A ] immediately after [ or [! is a literal
                i = 1
                negated = False
                if i < len(pattern) and pattern[i] == '!':
                    negated = True
                    i += 1
                # If ] is immediately after [ or [!, it's a literal ]
                if i < len(pattern) and pattern[i] == ']':
                    # This ] is a literal, so we need to find the real closing ]
                    i += 1
                    # Now look for the closing ]
                    while i < len(pattern) and pattern[i] != ']':
                        i += 1
                    if i >= len(pattern):
                        # Unterminated [ - treat [ as literal
                        if not name:
                            return False
                        if name[0] != '[':
                            return False
                        pattern = pattern[1:]
                        name = name[1:]
                        continue
                else:
                    # Look for the closing ]
                    while i < len(pattern) and pattern[i] != ']':
                        i += 1
                    if i >= len(pattern):
                        # Unterminated [ - treat [ as literal
                        if not name:
                            return False
                        if name[0] != '[':
                            return False
                        pattern = pattern[1:]
                        name = name[1:]
                        continue
                
                # Now pattern[i] is ']'
                # The class is pattern[1:i] (excluding the leading [ and trailing ])
                # But we need to handle the negation and the literal ] case
                # Let's re-parse more carefully
                
                # Actually, let me restructure this. Let me parse the class properly.
                # Start from index 1 (after [)
                j = 1
                negated = False
                if j < len(pattern) and pattern[j] == '!':
                    negated = True
                    j += 1
                
                # Check if the next char is ] (literal ])
                if j < len(pattern) and pattern[j] == ']':
                    # This ] is part of the class as a literal
                    j += 1
                    # Now find the closing ]
                    while j < len(pattern) and pattern[j] != ']':
                        j += 1
                    if j >= len(pattern):
                        # Unterminated
                        if not name:
                            return False
                        if name[0] != '[':
                            return False
                        pattern = pattern[1:]
                        name = name[1:]
                        continue
                else:
                    # Find the closing ]
                    while j < len(pattern) and pattern[j] != ']':
                        j += 1
                    if j >= len(pattern):
                        # Unterminated
                        if not name:
                            return False
                        if name[0] != '[':
                            return False
                        pattern = pattern[1:]
                        name = name[1:]
                        continue
                
                # Now pattern[j] is ']'
                # The class content is pattern[1:j]
                class_content = pattern[1:j]
                
                if not name:
                    return False
                
                ch = name[0]
                
                # Check if ch is in the class
                in_class = False
                k = 0
                while k < len(class_content):
                    if class_content[k] == '-' and k > 0 and k + 1 < len(class_content):
                        # Range
                        start = class_content[k-1]
                        end = class_content[k+1]
                        if start <= ch <= end:
                            in_class = True
                            break
                        k += 2
                    else:
                        if ch == class_content[k]:
                            in_class = True
                            break
                        k += 1
                
                if negated:
                    in_class = not in_class
                
                if not in_class:
                    return False
                
                pattern = pattern[j+1:]  # Skip past the ]
                name = name[1:]
            else:
                # Literal character
                if not name or name[0] != c:
                    return False
                pattern = pattern[1:]
                name = name[1:]
        
        return not name
    
    return match_here(pattern, name)
