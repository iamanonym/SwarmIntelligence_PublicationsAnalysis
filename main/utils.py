from main.const import FileConst
from PIL import ImageFont
import csv
import math
import itertools

# Precounted manually (for speed) list of top countries by impact on theme
# It is possibles to manipulate this list to get more or less concrete diagram
COUNTRIES = ['India', 'China', 'United Kingdom', 'United States', 'Australia',
             'South Korea', 'Saudi Arabia', 'Serbia', 'Iran', 'Italy',
             'Egypt']


def dist(point1, point2):
    """
        Count the distance between two points
    """
    x1, y1 = map(int, point1)
    x2, y2 = map(int, point2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def countIntersectCoordinates(x1, y1, x2, y2, xr, yr, rd):
    """
        Count the coordinates of intersection of segment and circle
    """
    l, r = 0, 1
    eps = 10 ** -6
    while r - l > eps:
        m = (l + r) / 2
        t = m / (1 - m)
        xm, ym = (x1 + t * x2) / (t + 1), (y1 + t * y2) / (t + 1)
        if (xm - xr) ** 2 + (ym - yr) ** 2 <= rd ** 2:
            l = m
        else:
            r = m
    t = l / (1 - l)
    xl, yl = (x1 + t * x2) / (t + 1), (y1 + t * y2) / (t + 1)
    return xl, yl


def normalize(x, y, k=1):
    """
        Transform vector to be unit vector
    """
    ln = (x ** 2 + y ** 2) ** 0.5
    if ln == 0:
        return 0, 0
    return x / ln * k, y / ln * k


def ellipseDivide(a, b, n):
    """
        Divide ellipse with a and b half-axis on n parts
    """

    def curve_len(t):
        return a * math.cos(t), b * math.sin(t)

    def integrate(f, a, b):
        ans = 0
        while a <= b:
            x1, y1 = f(a)
            x2, y2 = f(a + 0.01)
            ans += ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
            a += 0.01
        return ans

    points = [0]
    l, r, eps = 0, 2 * math.pi, 10 ** -3
    part = integrate(curve_len, l, r) / n
    for i in range(n - 1):
        l2, r2 = l, r
        while r2 - l2 > eps:
            m = (r2 + l2) / 2
            length = integrate(curve_len, l, m)
            if length <= part:
                l2 = m
            else:
                r2 = m
        l = l2
        points.append(l)
    return points


def isIntersect(group, segm):
    """
        Check if point group intersects the segment
    """
    point1, point2 = segm
    center = (group['x'], group['y'])
    a, b, c = dist(center, point1), dist(center, point2), dist(point1, point2)
    sab = (point1[0] - center[0]) * (point2[1] - center[1]) - (point2[0] - center[0]) * (point1[1] - center[1])
    h = abs(sab) / c
    if a < h or b < h or abs(math.sqrt(a ** 2 - h ** 2) + math.sqrt(b ** 2 - h ** 2) - c) > 1:
        return False
    return h <= group['r']


def areIntersect(segm1, segm2):
    """
        Check are two segments intersected
    """
    p1, p2 = segm1
    p3, p4 = segm2
    vsegm1 = (p2[0] - p1[0], p2[1] - p1[1])
    vsegm2 = (p4[0] - p3[0], p4[1] - p3[1])
    prod1 = (p3[0] - p1[0]) * vsegm1[1] - (p3[1] - p1[1]) * vsegm1[0]
    prod2 = (p4[0] - p1[0]) * vsegm1[1] - (p4[1] - p1[1]) * vsegm1[0]
    prod3 = (p1[0] - p3[0]) * vsegm2[1] - (p1[1] - p3[1]) * vsegm2[0]
    prod4 = (p2[0] - p3[0]) * vsegm2[1] - (p2[1] - p3[1]) * vsegm2[0]
    if math.copysign(1, prod1) != math.copysign(1, prod2) and math.copysign(1, prod3) != math.copysign(1, prod4):
        return True
    return False


def get_authors_by_file():
    """
        Get authors from csv-file
    """
    file = open(FileConst.TutorsList, encoding='utf-8')
    full_list = csv.reader(file, delimiter=',')
    ids_list, names_list, country_list = [], [], []
    for row in full_list:
        if row and row[0] != 'id':
            country = row[2].split('+')[0]
            if country not in COUNTRIES:
                continue
            ids_list.append(row[0])
            names_list.append(row[1])
            country_list.append(country)
    file.close()
    return ids_list, names_list, country_list


def get_top_writers(tutors_list, countries_list):
    """
        Filter authors with the most significant number of publications
    """
    file = open(FileConst.MainData, encoding='utf-8')
    full_list = csv.DictReader(file, delimiter=',')
    cols = full_list.fieldnames
    country_count = dict()
    pair_country_count = dict()
    article_by_pair_count = dict()
    for row in full_list:
        ids = [x.strip() for x in row[cols[2]].split(';')]
        for id in ids:
            for id2 in ids:
                if id != id2 and id in tutors_list and id2 in tutors_list:
                    edge = dict()
                    i, j = tutors_list.index(id), tutors_list.index(id2)
                    country_count[countries_list[i]] = country_count.get(countries_list[i], 0) + 1
                    country_count[countries_list[j]] = country_count.get(countries_list[j], 0) + 1
                    pair_country_count[(countries_list[i], countries_list[j])] =\
                        pair_country_count.get((countries_list[i], countries_list[j]), 0) + 1
                    pair_country_count[(countries_list[j], countries_list[i])] =\
                        pair_country_count.get((countries_list[j], countries_list[i]), 0) + 1
                    pair_country_count[(id, id2)] = pair_country_count.get((id, id2), 0) + 1
                    pair_country_count[(id2, id)] = pair_country_count.get((id2, id), 0) + 1
        for c1 in COUNTRIES:
            for c2 in COUNTRIES:
                if c1 in row[cols[15]] and c2 in row[cols[15]] and c1 != c2:
                    article_by_pair_count[(c1, c2)] = article_by_pair_count.get((c1, c2), 0) + 1
    file.close()
    return country_count, pair_country_count, article_by_pair_count


def build_country_graph(width, height):
    """
        Constructing the country graph
    """
    tutors_list, name_list, countries_list = get_authors_by_file()
    countries_unique = sorted(list(set(countries_list)))
    n, dn = len(tutors_list), len(countries_unique)
    vertex_list = []
    if n == 0:
        testfile = open('main/static/countries.svg', 'w')
        width = int(width)
        height = int(height)
        testfile.write('<svg width="{}" height="{}" viewBox="0 0 {} {}" '
                       'xmlns="http://www.w3.org/2000/svg"></svg>'.format(width, height,
                                                                    width, height))
        testfile.close()
        return testfile.name
    country_count, pair_country_count, article_by_pair_count =\
        get_top_writers(tutors_list, countries_list)
    ind = 0
    country_dict, tut_dict, name_dict = dict(), dict(), dict()
    for i in range(len(name_list)):
        name_dict[name_list[i]] = i
    for country in countries_unique:
        country_dict[country] = []
        for ind in range(len(name_list)):
            if country == countries_list[ind]:
                country_dict[country].append(tutors_list[ind])
    superTop = [country for country in country_count
                if country_count[country] > 1000]
    for country in superTop:
        countries_unique.remove(country)
    dn -= len(superTop)

    # CHANGE AUTHORS AND GROUPS ORDER

    for country in country_dict:
        for i in range(len(country_dict[country])):
            if pair_country_count.get((country_dict[country][i],
                                   country_dict[country][i - 1]), 0):
                j = (i + len(country_dict[country]) // 2) % len(country_dict[country])
                country_dict[country][i], country_dict[country][j] = \
                    country_dict[country][j], country_dict[country][i]
        if country in superTop:
            for i in range(len(country_dict[country])):
                if pair_country_count.get((country_dict[country][i],
                                           country_dict[country][i - 2]), 0):
                    j = (i + len(country_dict[country]) // 3) % len(country_dict[country])
                    country_dict[country][i], country_dict[country][j] = \
                        country_dict[country][j], country_dict[country][i]
    mxcntr = 0
    for perm in itertools.permutations(countries_unique):
        cntr = 0
        for i in range(1, len(perm)):
            cntr += pair_country_count.get((perm[i], perm[i - 1]), 0)
        cntr += pair_country_count.get((perm[0], perm[-1]), 0)
        if cntr > mxcntr:
            countries_unique = list(perm)
            mxcntr = cntr

    # DEFINE ELLIPSE DIVIDE ON IMAGE

    w0 = width / 2 - 7 * max(len(x) for x in name_list) - 10
    h0 = height / 2 - 45
    w, h = w0, h0
    cx, cy, dx, dy = 0, 0, 0, 0
    for u in range(5):
        angles = ellipseDivide(w, h, dn)
        l, r, m = 0.1, 1, 0.1
        while r - l > 0.001:
            m = (l + r) / 2
            ax, ay = (w * m, -w * m), (h * m, -h * m)
            for i in range(dn):
                x = math.cos(angles[i]) * w * m
                y = math.sin(angles[i]) * h * m
                rad = 80
                ax = (min(ax[0], x - rad), max(ax[1], x + rad))
                ay = (min(ay[0], y - rad), max(ay[1], y + rad))
            dx, dy = ax[1] - ax[0], ay[1] - ay[0]
            cx, cy = max(ax[1] - w0, 0), max(ay[1] - h0, 0)
            if dx > 2 * w0 or dy > 2 * h0:
                r = m
            else:
                l = m
        w, h = w * l * (2 * w0 / dx), h * l
    w /= (2 * w0 / dx)

    # FORMING VERTICES AND GROUPS

    vertex_list, group_list = [dict() for i in range(len(name_list))], []
    k, t = 0, 360 / dn
    cnt = 0
    for country in countries_unique:
        x = math.cos(angles[k]) * w + width / 2
        y = math.sin(angles[k]) * h + height / 2
        n2 = len(country_dict[country])
        rad = 70
        t2 = 360 / n2
        max_tg = 0.0000001
        ymin, ymax = 50000, 0
        for k2 in range(n2):
            y2 = math.sin(t2 * k2 / 180 * math.pi) * rad + y
            x2 = math.cos(t2 * k2 / 180 * math.pi) * rad + x
            if x2 > x:
                x2 += 10
            else:
                x2 -= 15
            ymin, ymax = min(ymin, y2), max(ymax, y2)
            max_tg = max(max_tg, abs((y2 - y) / (x2 - x)))
        for k2 in range(n2):
            y2 = math.sin(t2 * k2 / 180 * math.pi) * rad + y
            x2 = math.cos(t2 * k2 / 180 * math.pi) * rad + x
            x1, y1 = x2, y2 + 5
            fn = country_dict[country][k2]
            half = 0
            if x1 > x:
                x1 += 10
                half = 1
            else:
                x1 -= 15
                half = 2
            yprev, ynext = math.sin(t2 * (k2 - 1) / 180 * math.pi) * rad + y, \
                           math.sin(t2 * (k2 + 1) / 180 * math.pi) * rad + y
            if y2 > yprev and y2 > ynext or y2 < yprev and y2 < ynext:
                half = 0
                if y2 > y:
                    y1 += 3
            tg = min(abs((y2 - y) / (x1 - x)) ** 1.5, max_tg)
            if y2 < y:
                y1 -= 20 / max_tg * tg
            else:
                y1 += 15 / max_tg * tg
            number = ""
            r = 3
            if n2 > 100:
                r = 1
            elif n2 > 50:
                r = 2
            vertex_list[tutors_list.index(country_dict[country][k2])] = \
                {'x': x2, 'y': y2, 'x1': x1, 'y1': y1,
                 'number': number, 'country': country, 'r': r}
        group_list.append({'x': x, 'y': y, 'r': rad - 10, 'name': country})
        k, cnt = k + 1, cnt + n2
    for country in superTop:
        x = width / 2 + 200 * (-1) ** superTop.index(country)
        y = height / 2
        rad = 70
        n2 = len(country_dict[country])
        for k2 in range(n2):
            y2 = math.sin(t2 * k2 / 180 * math.pi) * rad + y
            x2 = math.cos(t2 * k2 / 180 * math.pi) * rad + x
            number = ""
            r = 3
            if n2 > 100:
                r = 1
            elif n2 > 50:
                r = 2
            vertex_list[tutors_list.index(country_dict[country][k2])] = \
                {'x': x2, 'y': y2, 'x1': x1, 'y1': y1,
                 'number': number, 'country': country, 'r': r}
        group_list.append({'x': x, 'y': y, 'r': rad - 10, 'name': country})
    countries_unique += superTop
    edge_list = []
    super_edge_set = set()

    # MAKING EDGES

    file = open(FileConst.MainData, encoding='utf-8')
    full_list = csv.DictReader(file, delimiter=',')
    cols = full_list.fieldnames
    for row in full_list:
        ids = [x.strip() for x in row[cols[2]].split(';')]
        for c1 in COUNTRIES:
            for c2 in COUNTRIES:
                if c1 in row[cols[15]] and c2 in row[cols[15]] and c1 != c2:
                    i, j = countries_unique.index(c1), \
                           countries_unique.index(c2)
                    v1, v2 = (group_list[i]['x'], group_list[i]['y']), \
                             (group_list[j]['x'], group_list[j]['y'])
                    point1 = countIntersectCoordinates(*v1, *v2, *v1, 70)
                    point2 = countIntersectCoordinates(*v2, *v1, *v2, 70)
                    flag = any(isIntersect(group, (point1, point2)) for group in group_list)
                    edge_list.append({'x1': point1[0], 'y1': point1[1],
                                      'x2': point2[0], 'y2': point2[1],
                                      'i': i, 'j': j, 'inter': True,
                                      'color': '#323cbd', 'arc': flag,
                                      'width': (article_by_pair_count[(c1, c2)] + 1) // 2})
        for id in ids:
            for id2 in ids:
                if id != id2 and id in tutors_list and id2 in tutors_list:
                    edge = dict()
                    i, j = tutors_list.index(id), tutors_list.index(id2)
                    if countries_list[i] == countries_list[j]:
                        v1, v2 = (vertex_list[i]['x'], vertex_list[i]['y']), \
                                 (vertex_list[j]['x'], vertex_list[j]['y'])
                        point1 = countIntersectCoordinates(*v1, *v2, *v1, 0)
                        point2 = countIntersectCoordinates(*v2, *v1, *v2, 0)
                        edge_list.append({'x1': point1[0], 'y1': point1[1],
                                          'x2': point2[0], 'y2': point2[1],
                                          'i': i, 'j': j, 'color': '#ff9d25',
                                          'inter': False, 'arc': False,
                                          'width': 1})
    file.close()

    # WRITE DOWN IN .SVG

    testfile = open('main/static/countries.svg', 'w')
    width = int(width)
    height = int(height)
    testfile.write('<svg width="{}" height="{}" viewBox="0 0 {} {}" '
                   'xmlns="http://www.w3.org/2000/svg">'.format(width, height,
                                                                width, height))
    isDrawn = dict()
    for edge in edge_list:
        if not edge['inter'] and countries_list[edge['i']] not in country_count:
            continue
        if not edge['inter'] and countries_list[edge['j']] not in country_count:
            continue
        if edge['arc']:
            if edge['x1'] > edge['x2']:
                edge['x1'], edge['x2'] = edge['x2'], edge['x1']
                edge['y1'], edge['y2'] = edge['y2'], edge['y1']
            midpoint = ((edge['x1'] + edge['x2']) / 2,
                        (edge['y1'] + edge['y2']) / 2)
            e = normalize(edge['x2'] - edge['x1'],
                          edge['y2'] - edge['y1'])
            ind1 = edge['i']
            ind2 = edge['j']
            if isDrawn.get((ind1, ind2), False):
                continue
            isDrawn[(ind1, ind2)] = True
            isDrawn[(ind2, ind1)] = True
            diff = 100 if abs(ind1 - ind2) < 3 else 170
            qpoint = (midpoint[0] - e[1] * diff,
                      midpoint[1] + e[0] * diff)
            testfile.write('<path d="M {} {} '
                           'Q {} {}, {} {}" '
                           'fill="transparent" '
                           'stroke="{}"/>'.format(edge['x1'], edge['y1'],
                                                  qpoint[0], qpoint[1],
                                                  edge['x2'], edge['y2'],
                                                  edge['color']))
        else:
            testfile.write('<line x1="{}" x2="{}"'
                           ' y1="{}" y2="{}" '
                           'stroke="{}" stroke-width="{}">'
                           '</line>'.format(edge['x1'], edge['x2'],
                                            edge['y1'], edge['y2'],
                                            edge['color'], edge['width']))
    for i in range(len(group_list)):
        x, y, r, name = group_list[i].values()
        x = int(x)
        y = int(y)
        testfile.write('<g><rect x="{}" y="{}" '
                       'width="{}" height="{}" fill="#E31E24"'
                       ' rx="5"></rect><text x="{}" y="{}" '
                       'style="fill: #FEFEFE; font-size: 14px" '
                       'text-anchor="middle">'
                       '{}</text></g>'.format(x - 5 * len(name), y,
                                              len(name) * 10, 18,
                                              x, y + 14, name))
    for i in range(len(vertex_list)):
        vertex = vertex_list[i]
        if countries_list[i] in country_count and country_count[countries_list[i]] > 200:
            testfile.write('<circle cx="{}" cy="{}" r="{}" '
                           'fill="{}"></circle>'.format(vertex['x'], vertex['y'],
                                                        vertex['r'], '#323cbd'))
    testfile.write('</svg>')
    testfile.close()
    return testfile.name


def build_keyword_graph(width, height):
    keywords_dict = dict()
    pair_keyword_dict = dict()
    file = open(FileConst.MainData, encoding='utf-8')
    full_list = csv.DictReader(file, delimiter=',')
    cols = full_list.fieldnames
    aliasDict = {"PSO": "Particle swarm optimization",
                 "ACO": "Ant colony optimization",
                 "Particle swarm optimization (pso)": "Particle swarm optimization",
                 "Wireless sensor network": "Wireless sensor networks"}
    for row in full_list:
        keywords = [x.strip() for x in row[cols[17]].split(';')]
        for i in range(len(keywords)):
            if not all(sym.isupper() for sym in keywords[i]):
                keywords[i] = keywords[i][0].upper() + keywords[i][1:].lower()
            keywords[i] = aliasDict.get(keywords[i], keywords[i])
        if '' in keywords:
            keywords.remove('')
        for keyword in keywords:
            keywords_dict[keyword] = keywords_dict.get(keyword, 0) + 1
            for keyword2 in keywords:
                if keyword != keyword2:
                    pair_keyword_dict[(keyword, keyword2)] = \
                        pair_keyword_dict.get((keyword, keyword2), 0) + 1
    keywords_list = [keyword for keyword in keywords_dict.keys()
                     if keywords_dict[keyword] > 15]
    superTop = [keyword for keyword in keywords_dict.keys()
                if keywords_dict[keyword] > 50]
    ultraTop = [keyword for keyword in keywords_dict.keys()
                if keywords_dict[keyword] > 200]
    for keyword in superTop:
        keywords_list.remove(keyword)
    for keyword in ultraTop:
        superTop.remove(keyword)
    group_dict = dict()
    for k1 in superTop:
        group = []
        for k2 in keywords_list:
            if pair_keyword_dict.get((k1, k2), 0):
                group_dict[k2] = k1
    for k1 in group_dict:
        for k2 in group_dict:
            g1, g2 = group_dict[k1], group_dict[k2]
            i1 = keywords_list.index(k1)
            i2 = keywords_list.index(k2)
            j1 = superTop.index(g1)
            j2 = superTop.index(g2)
            if abs(i1 - 6 * j1) + abs(i2 - 6 * j2) > \
                    abs(i1 - 6 * j2) + abs(i2 - 6 * j1):
                keywords_list[i1], keywords_list[i2] = keywords_list[i2], \
                                                       keywords_list[i1]
    full_keywords_list = keywords_list + superTop + ultraTop
    vertex_list = []
    rings = [keywords_list, superTop, ultraTop]
    colors = ['#622aba', '#323cbd', '#227cb1']
    max_str = max(len(x) + 5 for x in keywords_list) * 16 if keywords_list else 0
    rad0 = min(height - 110, width - max_str - 80) / 2
    cnt = 0
    for u in range(len(rings)):
        ring = rings[u]
        n = len(ring)
        if n == 0:
            continue
        rad = rad0 * (2 - cnt) // 2
        i, t = 0, 360 / n
        smr = 20 * (cnt + 2) // 2
        cnt += 1
        for i in range(n):
            x = math.cos(t * i / 180 * math.pi) * rad + width / 2
            y = math.sin(t * i / 180 * math.pi) * rad + height / 2
            x1, y1 = x, y
            if x > width / 2:
                x1 += smr + 5
            else:
                x1 -= smr + 5 + len(ring[i]) * 6
            if y1 > height / 2:
                y1 += smr + 18
            else:
                y1 -= smr + 8
            if abs(abs(y - height / 2) - rad) < 5:
                if x1 > width / 2:
                    x1 -= len(ring[i]) * 3
                else:
                    x1 += len(keywords_list[i]) * 3
            vertex_list.append({'x': x, 'y': y, 'number': keywords_dict[ring[i]],
                                'x1': x1, 'y1': y1, 'name': ring[i], 'r': smr,
                                'color': colors[u]})
    edge_list = []
    for i in range(len(full_keywords_list) - 1, 0, -1):
        for j in range(i - 1, 0, -1):
            k1, k2 = full_keywords_list[i], full_keywords_list[j]
            if pair_keyword_dict.get((k1, k2), 0) == 0:
                continue
            v1, v2 = (vertex_list[i]['x'], vertex_list[i]['y']), \
                     (vertex_list[j]['x'], vertex_list[j]['y'])
            point1 = countIntersectCoordinates(*v1, *v2, *v1, vertex_list[i]['r'] * 0.6)
            point2 = countIntersectCoordinates(*v2, *v1, *v2, vertex_list[j]['r'] * 0.6)
            color = colors[0]
            if vertex_list[i]['name'] in superTop or vertex_list[j]['name'] in superTop:
                color = colors[1]
            if vertex_list[i]['name'] in ultraTop or vertex_list[j]['name'] in ultraTop:
                color = colors[2]
            flag = True
            for u in range(len(full_keywords_list) - len(superTop) - len(ultraTop),
                           len(full_keywords_list) - len(ultraTop)):
                for w in range(u + 1, len(full_keywords_list) - len(ultraTop)):
                    point3 = (vertex_list[u]['x'], vertex_list[u]['y'])
                    point4 = (vertex_list[w]['x'], vertex_list[w]['y'])
                    if areIntersect((point1, point2), (point3, point4)):
                        flag = False
            if flag or vertex_list[i]['name'] in ultraTop or vertex_list[j]['name'] in ultraTop or \
                    (vertex_list[i]['name'] in superTop and vertex_list[j]['name'] in superTop):
                edge_list.append({'x1': point1[0], 'y1': point1[1],
                                  'x2': point2[0], 'y2': point2[1],
                                  'color': color, 'arc': False,
                                  'width': pair_keyword_dict[(k1, k2)]})
            elif not flag and \
                    int(vertex_list[i]['name'] in superTop) ^ int(vertex_list[j]['name'] in superTop):
                edge_list.append({'x1': vertex_list[i]['x'],
                                  'y1': vertex_list[i]['y'],
                                  'x2': vertex_list[j]['x'],
                                  'y2': vertex_list[j]['y'],
                                  'color': color, 'arc': True, 'top': True,
                                  'width': pair_keyword_dict[(k1, k2)]})
            else:
                edge_list.append({'x1': vertex_list[i]['x'],
                                  'y1': vertex_list[i]['y'],
                                  'x2': vertex_list[j]['x'],
                                  'y2': vertex_list[j]['y'],
                                  'color': color, 'arc': True,
                                  'width': pair_keyword_dict[(k1, k2)]})
    testfile = open('main/static/keywords.svg', 'w')
    width = int(width)
    height = int(height)
    testfile.write('<svg width="{}" height="{}" viewBox="0 0 {} {}" '
                   'xmlns="http://www.w3.org/2000/svg">'.format(width, height,
                                                                width, height))
    for edge in edge_list:
        if not edge['arc']:
            testfile.write('<line x1="{}" x2="{}"'
                           ' y1="{}" y2="{}" '
                           'stroke="{}" stroke-width="{}">'
                           '</line>'.format(edge['x1'], edge['x2'],
                                            edge['y1'], edge['y2'],
                                            edge['color'], edge['width']))
        elif edge.get('top', 0):
            midpoint = ((edge['x1'] + edge['x2']) / 2,
                        (edge['y1'] + edge['y2']) / 2)
            e = normalize(edge['x2'] - edge['x1'], edge['y2'] - edge['y1'],
                          k=rad0 * math.sqrt(3) / 2)
            angle = -math.copysign(1, e[0] * (height / 2 - edge['y1']) - e[1] * (width / 2 - edge['x1']))
            qx, qy = midpoint[0] - e[1] * angle, midpoint[1] + e[0] * angle
            testfile.write('<path d="M {} {} '
                           'Q {} {}, {} {}" '
                           'fill="transparent" '
                           'stroke="{}" '
                           'stroke-width="{}"/>'.format(edge['x1'], edge['y1'],
                                                        qx, qy, edge['x2'], edge['y2'],
                                                        edge['color'], edge['width']))
        else:
            midpoint = ((edge['x1'] + edge['x2']) / 2,
                        (edge['y1'] + edge['y2']) / 2)
            v = (midpoint[0] - width / 2, midpoint[1] - height / 2)
            e = normalize(*v, k=1.25 * rad0)
            qx, qy = width / 2 + e[0], height / 2 + e[1]
            testfile.write('<path d="M {} {} '
                           'Q {} {}, {} {}" '
                           'fill="transparent" '
                           'stroke="{}" '
                           'stroke-width="{}"/>'.format(edge['x1'], edge['y1'],
                                                        qx, qy, edge['x2'], edge['y2'],
                                                        edge['color'], edge['width']))
    font = ImageFont.truetype('times.ttf', 22)
    for i in range(len(vertex_list)):
        vertex = vertex_list[i]
        testfile.write('<circle cx="{}" cy="{}" r="{}" '
                       'fill="{}"></circle>'.format(vertex['x'], vertex['y'],
                                                    vertex['r'], vertex['color']))
        width = font.getlength(vertex['name']) + 30
        testfile.write('<g><rect x="{}" y="{}" '
                       'width="{}" height="{}" fill="#E31E24"'
                       ' rx="5"/><text x="{}" y="{}" text-anchor="middle" '
                       'fill="#ffffff" font-size="18pt">{}'
                       '</text></g>'.format(vertex['x1'], vertex['y1'] - 20,
                                            width, 24,
                                            vertex['x1'] + width // 2,
                                            vertex['y1'],
                                            vertex['name']))
    testfile.write('</svg>')
    testfile.close()
    return testfile.name
