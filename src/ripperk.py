"""
Python implementation of the the RIPPERk algorithm.

Fast Effective Rule Induction
William W. Cohen
AT&T Bell Laboratories
600 Mountain Avenue Murray Hill, NJ 07974
wcohen@research.att.com
@see http://www.cs.fit.edu/~pkc/ml/related/cohen-icml95.pdf

The style guide follows the strict python PEP 8 guidelines.
@see http://www.python.org/dev/peps/pep-0008/

@author Aaron Zampaglione <azampagl@my.fit.edu>
@course CSE 5800 Advanced Topics in CS: Learning/Mining and the Internet, Fall 2011
@project Proj 01, RIPPERk
"""
import getopt
import math
import sys

#
# Global META variable.
#
# Normally, globals are bad.  However,
# the code is built for performance
# and therefore avoids objects.
#
# Once concise dictionary for misc. items.
#
meta = {
        # The options dictionary contains all the user defined arguments.
        'opts': {},
        # The attribute dictionary contains all the attributes found in the
        #  attribute file.
        'attrs': {},
        # The total possible attribute combinations; a.k.a the "n" of MDL.
        'n': 0,
        # Bit length threshold
        'd': 64,
        # Growing to pruning set ratio.
        'ratio': 2 / float(3),
        }

def bindings(cases, rules):
    """
    Finds the number of bindings to a training set on
    a given number of rules.
    
    Put simply, it determines intersections between
    dictionaries.
    
    Each item in the training set must bind to each rule
    if more than one is given in the rules variable.
    
    Key arguments:
    cases -- the training set (cases) to look at.
    rules -- the rules analyze.
    """
    count = 0
    for case in cases:
        
        rules_success = True
        
        for rule in rules:
            
            rule_success = True
            
            for attr, value in rule.iteritems():
                if attr in case:
                    
                    attr_success = False
                    
                    for case_value in case[attr]:
                        if value[0] == "==":
                            attr_success = value[1] == case_value
                        elif value[0] == ">=":
                            attr_success = value[1] <= case_value
                        elif value[0] == "<=":
                            attr_success = value[1] >= case_value
                    
                    if not attr_success:
                        rule_success = False
                else:
                    rule_success = False
                
            if not rule_success:
                rules_success = False
        
        if rules_success:
            count += 1
    
    return count

def classify(classes):
    """
    Classifies given test cases.
    
    Key arguments:
    classes -- the testing cases to classify.
    """
    f = open(meta['opts']['m'], 'r')
    # Dangerous but we're not interested in security right now.
    model = eval(f.readline())
    f.close()
    
    # Our default class is the only one not found in the model.
    default = set(classes.keys()).difference(model.keys()).pop()
    
    # Keep track of the results for each case.
    results = {default: (len(classes[default]), 0)}
    
    # Sort in order
    items = classes.items()
    items.sort(key=lambda i: len(i[1]))
    
    # Loop over every test case and attempt to classify.
    for c, cases in items:
        
        # Init the results for this class.
        results[c] = (len(cases), 0)
        
        for case in cases:
            
            # This case was satisfied by one of the non-default classes.
            found = False
            
            for m, ruleset in model.iteritems():
                
                # No need to continue of we found a rule that satisfies the case.
                if found:
                    break
                
                for rule in ruleset:
                    if bindings([case], [rule]):
                        if c == m:
                            results[c] = (results[c][0], results[c][1] + 1)
                        found = True
                        break
            
            # The case wasn't satisfied by any of the rulesets.
            # Check if it was our default case.
            if not found and c == default:
                results[default] = (results[default][0], results[default][1] + 1)
    
    total = 0
    classified = 0
    
    # Output our results to the result file.
    output = "Class\t\t\tCases\t\t\tClassified\n\n"
    for attr, result in results.iteritems():
        total += result[0]
        classified += result[1]
        output += attr + "\t\t\t" + str(result[0]) + "\t\t\t" + str(result[1]) + "\n"
    output += "\n"
    output += "Accuracy: " + str(classified * 100 / float(total)) + "%"
    f = open(meta['opts']['o'], 'w')
    f.write(output)
    f.close()
    
    # Dirty trick to capture output for our corrupter.
    print(str(classified * 100 / float(total)))
    
def create_attrs(f):
    """
    Parses an attribute file into an attribute dictionary.
    
    Each key in the dictionary (attribute) contains a tuple 
    that has the attribute's index in the file (line number), 
    wheter the attribute is continuous or discrete (True|False), 
    and a dictionary of possible values (each value records 
    the number of times that value was seen.  All are initialized 
    to zero).
    
    e.g.:
    
        attrs['Bar'] = (1, False, {'Yes': 0, 'No': 0})
    
    Key arguments:
    f -- the attribute file handle.
    """
    attrs = {}
    
    i = 0
    for line in f:
        # The first item is the attribute name, the rest are possible values.
        split = line[:-1].split(" ")
        attr = split[0]
        values = split[1:]
        # Continuous case.
        if len(values) == 1 and values[0].lower() in ['int', 'float', 'long', 'complex']:
            attrs[attr] = (i, values[0].lower(), {})
        # Discrete case.
        else:
            v = {}
            for value in values:
                v[value] = 0
            attrs[attr] = (i, False, v)
        
        i += 1
    
    return attrs

def create_classes(f):
    """
    Creates the classes dictionary.
    
    The classes dictionary is more or less our training set.  Each
    key in the dictionary (class) contains the corresponding training
    cases.
    
    Each training case is a dictionary where the key is the attribute
    name and the value the possible values.
    
    e.g.:
    
        If we are trying to predict WillWait, our classes dictionary would look
        like...
    
        ['Yes'] = [{
                    'Alternate': ['Yes']
                    'Bar': ['No']
                    },
                   ...
                  ]
        
        ['No'] = ...
    
    Key arguments:
    f  -- the file handle.
    """
    # Classes dictionary.
    classes = {}
    
    # Build a reverse lookup dictionary for attribute index values.
    indices = {}
    for attr, value in meta['attrs'].iteritems():
        indices[value[0]] = attr
    
    for line in f:
        case = {}
        # Index counter.
        i = 0
        split = line[:-1].split(" ")
        for value in split:
            # Find the attr name.
            attr = indices[i]
            # Check if this is the defining attribute, skip if so.
            if attr != meta['opts']['c']:
                # Check for continuous.
                if meta['attrs'][attr][1]:
                    # This is a dangerous cast!!  But we made sure it was a numeric 
                    #  type earlier!!
                    value = eval(meta['attrs'][attr][1])(value)
                    
                    if attr in case:
                        case[attr].append(value)
                    else:
                        case[attr] = [value]
                    
                    if value in meta['attrs'][attr][2]:
                        meta['attrs'][attr][2][value] += 1
                    else:
                        meta['attrs'][attr][2][value] = 0
                else:
                    # Set the attribute and the value.
                    case[attr] = [value]
                    
                    meta['attrs'][attr][2][value] += 1
            else:
                c = value
                meta['attrs'][attr][2][value] += 1
            
            i += 1
        
        # Create a new class if necessary.
        if not c in classes:
            classes[c] = []
        
        # Append the case to our test set.
        classes[c].append(case)
    
    # Return classes dictionary.
    return classes

def dl(rule):
    """
    Finds the description length for a rule.
    
    Key arguments:
    rule -- the rule.
    """
    k = len(rule.keys())
    p = k / float(meta['n'])
    
    p1 = float(k) * math.log(1 / p, 2)
    p2 = float(meta['n'] - k) * math.log(1 / float(1 - p), 2)
    
    return int(0.5 * (math.log(k, 2) + p1 + p2))
    
def foil(pos, neg, lit, rule, rules):
    """
    FOIL-GAIN
    
    Key arguments:
    pos   -- the positive cases.
    neg   -- the negative cases.
    lit   -- the literal to add.
    rule  -- the rule.
    rules -- the rules to compare to.
    """
    # Append our rule to the rule list.
    rules.append(rule)
    
    if rule:
        p0 = bindings(pos, rules)
        n0 = bindings(neg, rules)
    else:
        p0 = len(pos)
        n0 = len(neg)
    
    # Remove rule from stack.
    rules.pop()
    
    # Make a new rule with the literal.
    new_rule = dict(rule)
    new_rule[lit[0]] = lit[1]
    
    # Append the new rule and test bindings.
    rules.append(new_rule)
    
    p1 = bindings(pos, rules)
    n1 = bindings(neg, rules)
    
    # Remove new rule from stack.
    rules.pop()
    
    # Check for division by 0.
    if p0 == 0:
        d0 = 0
    else:
        d0 = math.log(float(p0) / (float(p0) + float(n0)), 2)
    
    if p1 == 0:
        d1 = 0
    else:
        d1 = math.log(float(p1) / (float(p1) + float(n1)), 2)
    
    # Pop both rules on stack.
    rules.append(rule)
    rules.append(new_rule)
    
    t = bindings(pos, rules)
    
    # Pop both rules off of stack.
    rules.pop()
    rules.pop()
    
    return t * (d1 - d0)

def grow_rule(pos, neg, rule=None, rules=None):
    """
    Grows (adds conditions) to a rule until it matches no negative cases.
    
    Key arguments
    pos   -- the positive training cases.
    neg   -- the negative training cases.
    rule  -- [optional] the rule to grow
    rules -- [optional] the rules to bind to
    """
    if rule == None:
        rule = {}
    if rules == None:
        rules = []
        
    while True:
        max_gain = None
        max_condition = None
        
        for attr, values in meta['attrs'].iteritems():
            
            # Can't add an attribute twice.
            if attr in rule:
                continue
            
            # Conditions for this attribute.
            conditions = []
            
            # Continuous.
            if values[1]:
                for v in values[2].keys():
                    conditions.append((attr, (">=", v)))
                    conditions.append((attr, ("<=", v)))
            # Discrete.
            else:
                for value in values[2].keys():
                    conditions.append((attr, ("==", value)))
                
            # Check the gain for each condition.
            for condition in conditions:
                gain = foil(pos, neg, condition, rule, rules)
                if max_gain < gain:
                    max_condition = condition
                    max_gain = gain
        
        # Add the new max condition.
        rule[max_condition[0]] = max_condition[1]
        rules.append(rule)
        
        # Check if it covers no negative cases.
        if max_gain <= 0.0 or bindings(neg, rules) == 0:
            rules.pop()
            return rule
        else:
            rules.pop()

def irep(pos, neg):
    """
    IREP
    
    Key arguments:
    pos -- the positive cases.
    neg -- the negative cases.
    """
    # The final rule set.
    rule_set = []
    
    # The minimum dl recorded.
    min_dl = meta['n']
    
    while pos:
        pos_len = len(pos)
        neg_len = len(neg)
        
        if meta['opts']['p']:
            ratio = 1
        else:
            ratio = meta['ratio']
        
        pos_chunk = int(pos_len * ratio)
        neg_chunk = int(neg_len * ratio)
        
        if pos_chunk == 0:
            pos_chunk = 1
            
        if neg_len > 1 and neg_chunk == 0:
            neg_chunk = 1
        
        # Grow.
        grow_pos = pos[:pos_chunk]
        grow_neg = neg[:neg_chunk]
        rule = grow_rule(grow_pos, grow_neg)
        
        # Prune.
        if meta['opts']['p']:
            prune_pos = pos[pos_chunk:]
            prune_neg = neg[neg_chunk:]
            rule = prune_rule(prune_pos, prune_neg, rule)
        
        rule_dl = dl(rule)
        
        if min_dl + meta['d'] < rule_dl:
            return rule_set
        else:
            # Add our rule to the ruleset.
            rule_set.append(rule)
            
            # Reset mindl, if necessary.
            if rule_dl < min_dl:
                min_dl = rule_dl
            
            # Remove the cases found by rule.
            pos = remove_cases(pos, rule)
            neg = remove_cases(neg, rule)
    
    return rule_set

def learn(classes):
    """
    Learns the classes and outputs to model and output file.
    
    Key arguments:
    classes -- the classes to learn.
    """
    # Sort our classes in order of least prevalence.  Because python works
    #  with lists backwards, it is going to run in reverse
    items = classes.items()
    items.sort(key=lambda i: len(i[1]), reverse=True)
    
    # Get all the classes.
    attrs = classes.keys()
    
    # The rule sets learned.
    rulesets = {}
    
    # Clean output
    output = ""
    
    while len(items) > 1:
        # Get the current class.
        c = items.pop()
        # The positive training set is the current classes training cases.
        pos = []
        # The negative training set is the remainder cases.
        neg = []
        for attr in attrs:
            # Deep copy this classes training cases (positive).
            if attr == c[0]:
                pos = list(classes[attr])
            # Deep copy all other training cases (negative).
            else:
                neg.extend(list(classes[attr]))
        
        # Build the ruleset.
        ruleset = irep(pos, neg)
        
        # Optimize k times.
        for _ in range(meta['opts']['k']):
            ruleset = optimize(pos, neg, ruleset)
        
        # Remove cases covered by rule.
        for attr in attrs:
            for rule in ruleset:
                classes[attr] = remove_cases(classes[attr], rule)
        
        rulesets[c[0]] = ruleset
    
        str_ruleset = ""
        for rule in ruleset:
            str_rule = ""
            for attr, value in rule.iteritems():
                str_rule += str(attr) + " " + str(value[0]) + " " + str(value[1]) + " && "
            str_ruleset += str_rule[:-4] + " || "
        
        output += "IF " + str_ruleset[:-4] + " THEN " + str(c[0]) + "\n"
    
    # Write the model to file
    f = open(meta['opts']['m'], 'w')
    f.write(str(rulesets))
    f.close()
    
    c = items.pop()
    output += "ELSE " + c[0]
    
    # Write the model to file
    f = open(meta['opts']['o'], 'w')
    f.write(str(output))
    f.close()

def main():
    """Main execution method."""
    # Determine command line arguments
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "e:a:c:t:m:o:k:p:")
    except getopt.GetoptError, err:
        print(err)
        usage()
        sys.exit(2)
    
    # Process each command line argument.
    for o, a in opts:
        meta['opts'][o[1]] = a
    
    # The following arguments are required in all cases.
    for opt in ['e', 'a', 'c', 't', 'm', 'o']:
        if not opt in meta['opts']:
            usage()
            sys.exit(2)
    
    # Execute option can be only one of two types.
    if meta['opts']['e'] != 'learn' and meta['opts']['e'] != 'classify':
        usage()
        sys.exit(2)
    
    # Special operations.
    if 'k' in meta['opts']:
        meta['opts']['k'] = int(meta['opts']['k'])
    else:
        meta['opts']['k'] = 2
    if 'p' in meta['opts']:
        meta['opts']['p'] = bool(int(meta['opts']['p']))
    else:
        meta['opts']['p'] = True
    
    # Create attrs.
    f = open(meta['opts']['a'])
    meta['attrs'] = create_attrs(f)
    f.close()
    
    # Create classes.
    f = open(meta['opts']['t'])
    classes = create_classes(f)
    f.close()
    
    # No need to keep track of our predict attr at this point.
    del meta['attrs'][meta['opts']['c']]
    
    # Remove attributes that we have never seen in our training set.
    for attr in meta['attrs'].keys():
        # Only look at discrete attributes.
        if not meta['attrs'][attr][1]:
            for value in meta['attrs'][attr][2].keys():
                if meta['attrs'][attr][2][value] == 0:
                    del meta['attrs'][attr][2][value]
    
    if meta['opts']['e'] == 'learn':
        # Determine the total possible attribute combinations.
        #  a.k.a the "n" of MDL.
        for attr, values in meta['attrs'].iteritems():
            # Continuous.
            if values[1]:
                meta['n'] += 2 * len(values[2].keys())
            # Discrete.
            else:
                meta['n'] += 1
        
        learn(classes)
    else:
        classify(classes)

def optimize(pos, neg, ruleset):
    """
    Globally optimizes the rule set.
    
    Key arguments:
    pos     -- the positive training cases.
    neg     -- the negative training cases.
    ruleset -- the ruleset.
    """
    new_ruleset = list(ruleset)
    
    i = 0
    while i < len(new_ruleset):
        rule = new_ruleset.pop(i)
        
        # Grow a new rule that might replace the old one.
        reprule = grow_rule(pos, neg, rules=new_ruleset)
        if meta['opts']['p']:
            reprule = prune_rule(pos, neg, reprule, new_ruleset)
            
        # Grow without pruning our original rule.
        revrule = grow_rule(pos, neg, dict(rule), new_ruleset)
        
        # Get the description length for each rule.
        rule_dl = dl(rule)
        reprule_dl = dl(reprule)
        revrule_dl = dl(revrule)
        
        if (reprule_dl < rule_dl and reprule_dl < revrule_dl):
            # Don't allow duplicates.
            if not reprule in new_ruleset:
                new_ruleset.insert(i, reprule)
        elif (revrule_dl < rule_dl and revrule_dl < reprule_dl):
            # Don't allow duplicates.
            if not revrule in new_ruleset:
                new_ruleset.insert(i, revrule)
        else:
            # Don't allow duplicates.
            if not rule in new_ruleset:
                new_ruleset.insert(i, rule)
        
        i+= 1
        
    return new_ruleset

def prune_rule(pos, neg, rule, rules=None):
    """
    Prunes a rule.
    
    Key arguments:
    pos   -- the positive cases.
    neg   -- the negative cases.
    rule  -- the rule to prune.
    rules -- [optional] rules to bind to.
    """
    if rules == None:
        rules = []
    
    # Deep copy our rule.
    tmp_rule = dict(rule)
    # Append the rule to the rules list.
    rules.append(rule)
    
    p = bindings(pos, rules)
    n = bindings(neg, rules)
    
    if p == 0 and n == 0:
        rules.pop()
        return tmp_rule
    
    max_rule = dict(tmp_rule)
    max_score = (p - n) / float(p + n)
    
    keys = max_rule.keys()
    i = -1
    
    while len(tmp_rule.keys()) > 1:
        # Remove the last attribute.
        del tmp_rule[keys[i]]
        
        # Recalculate score.
        p = bindings(pos, rules)
        n = bindings(neg, rules)
        
        if p == 0 and n == 0:
            continue
        
        tmp_score = (p - n) / float(p + n)
        
        # We found a new max score, save rule.
        if tmp_score > max_score:
            max_rule = dict(tmp_rule)
            max_score = tmp_score
    
        i -= 1
    
    # Remove the rule from the rules list.
    rules.pop()
    
    return max_rule
    
def remove_cases(cases, rule):
    """
    Removes cases based on a rule.
    
    Key arguments:
    cases -- the cases to analyze.
    rule  -- the rule to analyze with.
    """
    new_cases = []
    for case in cases:
        if not bindings([case], [rule]):
            new_cases.append(case)
    
    return new_cases

def usage():
    """Prints the usage of the program."""
    print("\n" + 
          "The following are arguments required:\n" + 
          "-e: the execution method (learn|classify)\n" + 
          "-a: the attribute file location.\n" + 
          "-c: the defining attribute in the attribute file (a.k.a what we are trying to predict).\n" + 
          "-t: the training/testing file location.\n" + 
          "-m: the model file (machine readable results).\n" + 
          "-o: the output file (human readable results).\n" + 
          "\n" + 
          "The following are arguments are optional\n" + 
          "-k: the number of optimizations (default is 2)\n" +
          "-p: pruning or no pruning [1 or 0] (default is 1)\n" + 
          "\n" + 
          "Example Usage:\n" + 
          "python ripperk.py -e learn -a \"../data/restaurant-attr.txt\"" + 
          " -c WillWait -t \"../data/restaurant-train.txt\" -m \"model.dat\"" + 
          " -o \"train-results.txt\" -k 2 -p 1" + 
          "\n" + 
          "python ripperk.py -e classify -a \"../data/restaurant-attr.txt\"" + 
          " -c WillWait -t \"../data/restaurant-text.txt\" -m \"model.dat\"" + 
          " -o \"test-results.txt\"" + 
          "\n")

"""Main execution."""
if __name__ == "__main__":
    main()