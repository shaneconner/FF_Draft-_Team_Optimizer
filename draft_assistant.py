import pandas as pd
import numpy as np
import itertools
from scipy import stats
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


def get_ffa(file_location='./data/ffa_customrankings2017-0.csv'):

    # Subset within ADP/VOR/ECR of 160 (drafted within a typical ESPN draft, 10 teams, 16 rounds)
    df = pd.read_csv(file_location)

    # Subset columns
    df = df[['overallRank',
            'player',
            'team',
            'playerposition',
            'points',
            'lower',
            'upper',
            'bye',
            'positionRank',
            'playerId',
            ]]

    # Remove periods from player name for easier merges
    df['player'] = df['player'].str.replace('.','')
    df['player'] = df['player'].str.title()

    # Swap defense player name for easier merges
    df.ix[df['playerposition'] == 'DST', 'player'] = df['team'] + " " + "DEF"

    # Sort by value over replacement rank and use as index
    df = df.sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)

    return df


# Gather ADP (average draft position) from actual drafts from
# https://fantasyfootballcalculator.com/adp?format=standard&year=2017&teams=10&view=graph&pos=all
def get_adp(file_location='./data/adp.csv'):

    # Read in ADP data
    df = pd.read_csv(file_location)
    # Isolate first name and replace periods for cleaner merge
    df['first_name'] = df['Name'].str.split(' ').str[0].map(lambda x: x.replace('.', ''))
    # Isolate last name and replace periods for cleaner merge
    df['last_name'] = df['Name'].str.split(' ').str[1].map(lambda x: x.replace('.', ''))
    # Create new column on name
    df['player'] = df['first_name'] + " " + df['last_name']
    df['player'] = df['player'].str.title()
    # Create new column on name
    df.ix[df['Pos'] == 'DEF', 'player'] = df['Team'] + " " + "DEF"
    # Rename ADP
    df = df.rename(columns = {'Overall':'ADP'})

    # Subset columns
    df = df[['ADP',
            'player',
            'Pos',
            'Std.'
            ]]

    return df


def get_schedule():

    # schedule - Load
    qb_schedule = pd.read_csv('./data/FantasyPros_Fantasy_Football_2017_QB_Matchups.csv')
    rb_schedule = pd.read_csv('./data/FantasyPros_Fantasy_Football_2017_RB_Matchups.csv')
    wr_schedule = pd.read_csv('./data/FantasyPros_Fantasy_Football_2017_WR_Matchups.csv')
    te_schedule = pd.read_csv('./data/FantasyPros_Fantasy_Football_2017_TE_Matchups.csv')
    k_schedule = pd.read_csv('./data/FantasyPros_Fantasy_Football_2017_K_Matchups.csv')
    dst_schedule = pd.read_csv('./data/FantasyPros_Fantasy_Football_2017_DST_Matchups.csv')

    # Concatenate all schedules
    schedules = [qb_schedule, rb_schedule, wr_schedule, te_schedule, k_schedule, dst_schedule]
    schedule = pd.concat(schedules)

    # Remove periods from player name for easier merges
    schedule['Player'] = schedule['Player'].str.replace('.','')
    schedule['Player'] = schedule['Player'].str.replace("'","")
    schedule['Player'] = schedule['Player'].str.title()


    # Week columns
    week_columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',                 '11', '12', '13', '14', '15', '16', '17']

    # Change "BYE" to 0
    schedule[week_columns] = schedule[week_columns].replace("BYE", float(0))

    # Sum schedules points allowed for position
    schedule['total'] = schedule[week_columns].sum(axis=1)

    # Isolate first name and replace periods for cleaner merge
    schedule['Player'] = schedule['Player'].astype(str)
    schedule['first_name'] = schedule['Player'].str.split(' ').str[0].map(lambda x: x.replace('.', ''))
    # Isolate last name and replace periods for cleaner merge
    schedule['last_name'] = schedule['Player'].str.split(' ').str[1].astype(str).map(lambda x: x.replace('.', ''))
    # Create new column on name
    schedule['player'] = schedule['first_name'] + " " + schedule['last_name']

    # Remove unnecessary columns
    schedule = schedule.drop('ECR', 1)
    schedule = schedule.drop('first_name', 1)
    schedule = schedule.drop('last_name', 1)
    schedule = schedule.drop('Player', 1)

    # Change weekly score to proportion of points allowed
    for week in week_columns:
        schedule[week] = schedule[week].astype(float) / schedule['total']

    return schedule


def weekly_projections(df, points):

    # Subset dataframe to rows containing non NaN values for points & bye weeks
    df = df[np.isfinite(df[points])]
    df = df[np.isfinite(df['bye'])]

    # Schedule
    schedule = get_schedule()

    # Merge ffa and adp dataframes together
    df = pd.merge(df, schedule, on='player', how='left')

    # Create a variable for each week's (1-17) and add a projected weekly score
    for i in range(1,18):
        column_name = "week_" + str(i) + "_" + str(points)
        df[column_name] = df[points] * df[str(i)]
        df = df.drop(str(i), 1)

    # Drop total
    df = df.drop('total', 1)

    return df


# Merge datasets together
def player_data(ffa_file='./data/ffa_customrankings2017-0.csv', adp_file='./data/adp.csv', rounds=16, teams=10):

    # Grab player projections from Fantasy Football Analytics CSV
    ffa = get_ffa(ffa_file)
    # Get ADP data from Fantasy Football Calculator CSV
    adp = get_adp(adp_file)

    # Merge ffa and adp dataframes together
    df = pd.merge(ffa, adp, on='player', how='left')

    # Add weekly projections for points
    df = weekly_projections(df, 'points')
    # Add weekly projections for lower points
    df = weekly_projections(df, 'lower')
    # Add weekly projections for upper points
    df = weekly_projections(df, 'upper')

    # Subset field
    adp = adp[adp['ADP'] < int(rounds * teams)]
    qb = int(len(adp[adp['Pos'] == 'QB']) * 1.25)
    rb = int(len(adp[adp['Pos'] == 'RB']) * 1.25)
    wr = int(len(adp[adp['Pos'] == 'WR']) * 1.25)
    te = int(len(adp[adp['Pos'] == 'TE']) * 1.25)

    df = df.query("playerposition !='QB' | positionRank < " + str(qb))
    df = df.query("playerposition !='RB' | positionRank < " + str(rb))
    df = df.query("playerposition !='WR' | positionRank < " + str(wr))
    df = df.query("playerposition !='TE' | positionRank < " + str(te))
    df = df.query("playerposition !='DST' | positionRank < " + str((teams * 1.25)))
    df = df.query("playerposition !='K' | positionRank < " + str(teams * 1.25))

    # Drop unnecessary columns
    df = df.drop('Pos', 1)
    df = df.drop_duplicates(subset='player', keep='first')

    # Sort by value over replacement rank and use as index
    df = df.sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)

    return df


# Creates new column that takes a player's projected points and takes
# the difference from the median of the rest of the field
def value_over_replacement(df):

    # Difference between player's projected points vs median projected points
    df['avg_value_over_replacement'] = df['points'] - np.nanmedian(df['points'])
    # Difference between player's projected lower points vs median projected lower points
    df['lower_value_over_replacement'] = df['lower'] - np.nanmedian(df['lower'])
    # Difference between player's projected upper points vs median projected upper points
    df['upper_value_over_replacement'] = df['upper'] - np.nanmedian(df['upper'])

    return df


# Survival probability of player for next pick
def survival(df, next_pick):

    # Using x as the next pick, ADP as loc (mean), Std. as scale (standard deviation)
    df['survival_probability'] = stats.norm.sf(x=next_pick,                                                loc=df['ADP'],                                                scale= df['Std.'])

    # Round and convert to percentage for ease of comprehension
    df['survival_probability'] = 100 * df['survival_probability'].round(6)

    return df


def add_features(df, pick, next_pick):

    # Create a variable which measures a player's points over median points relative to their position
    df = df.groupby(['playerposition']).apply(value_over_replacement)

    # Rerank ADP based on existing picks
    df['ADP'] = df['ADP'].rank(ascending=True) + pick - 1

    # Create a variable which measures a player's projected points zscore relative to their position
    df['avg_points_zscore'] = df.groupby(['playerposition'])['points'].transform(stats.zscore)
    # Create a variable which measures a player's lower projected points zscore relative to their position
    df['lower_points_zscore'] = df.groupby(['playerposition'])['lower'].transform(stats.zscore)
    # Create a variable which measures a player's upper projected points zscore relative to their position
    df['upper_points_zscore'] = df.groupby(['playerposition'])['upper'].transform(stats.zscore)

    # Create a variable which measures a player's probability of availability for user's next draft pick
    df = survival(df, next_pick)

    return df


def top_players(df, roster):

    players = []

    if len(roster[roster['playerposition'] == 'QB']) < 2:
        position = df[df['playerposition'] == 'QB'].sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)
        players += position.values.tolist()[:(3 - len(roster[roster['playerposition'] == 'QB']))]

    if len(roster[roster['playerposition'] == 'RB']) < 6:
        position = df[df['playerposition'] == 'RB'].sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)
        players += position.values.tolist()[:(6 - len(roster[roster['playerposition'] == 'RB']))]

    if len(roster[roster['playerposition'] == 'WR']) < 6:
        position = df[df['playerposition'] == 'WR'].sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)
        players += position.values.tolist()[:(6 - len(roster[roster['playerposition'] == 'WR']))]

    if len(roster[roster['playerposition'] == 'TE']) < 2:
        position = df[df['playerposition'] == 'TE'].sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)
        players += position.values.tolist()[:(3 - len(roster[roster['playerposition'] == 'TE']))]

    return players


def grab_special(df, roster):

    players = []

    if len(roster[roster['playerposition'] == 'K']) < 1:
        position = df[df['playerposition'] == 'K'].sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)
        position['st_score'] = 0
        position['bench'] = position.points.diff(-1)
        position['low_st_score'] = 0
        position['low_bench'] = position.points.diff(-1)
        position['high_st_score'] = 0
        position['high_bench'] = position.points.diff(-1)
        position_list = position.values.tolist()[:(3 - len(roster[roster['playerposition'] == 'K']))]
        players += position_list

    if len(roster[roster['playerposition'] == 'DST']) < 1:
        position = df[df['playerposition'] == 'DST'].sort_values(by=['overallRank'], ascending=True).reset_index(drop=True)
        position['st_score'] = 0
        position['bench'] = position.points.diff(-1)
        position['low_st_score'] = 0
        position['low_bench'] = position.points.diff(-1)
        position['high_st_score'] = 0
        position['high_bench'] = position.points.diff(-1)
        position_list = position.values.tolist()[:(3 - len(roster[roster['playerposition'] == 'DST']))]
        players += position_list

    return players


def make_teams(players, roster):

    # Convert roster to list format to allow merge with each team iteration
    my_roster = roster[(roster['playerposition'] != 'DST') & (roster['playerposition'] != 'K')]
    roster_size = len(my_roster)
    # Create team combinations of all top available player (N of Players) choose 14
    teams = list(itertools.combinations(players, 14 - roster_size))
    # Remove invalid teams from list of teams
    valid_teams = validate_teams(teams=teams, roster=my_roster.values.tolist())

    return valid_teams


def validate_teams(teams, roster):

    valid_teams = []
    i = 0
    # Iterate through all teams
    while i < len(teams):

        # Add roster to team
        teams[i] = roster + list(teams[i])

        # Count number of positions per team
        counts = Counter(x for x in list(itertools.chain.from_iterable(teams[i])))

        # Remove teams if there are too many or too little of any position
        if counts['QB'] != 2         or counts['RB'] > 6         or counts['WR'] > 6         or counts['TE'] != 2         or counts['RB'] < 4         or counts['WR'] < 4:
            del teams[i]

        # If valid, score the team's starters and backups
        else:

            score = []
            low_score = []
            high_score = []

            # Retrieve a player's projected weekly score
            for week in range(17):
                # Standard score
                score.append(score_week(teams[i], week + 12))
                # Lower score
                low_score.append(score_week(teams[i], week + 29))
                # Upper score
                high_score.append(score_week(teams[i], week + 46))

            # Add scores to teams
            teams[i].append([sum(starter[0] for starter in score),                             sum(bench[1] for bench in score)])

            teams[i].append([sum(starter[0] for starter in low_score),                             sum(bench[1] for bench in low_score)])

            teams[i].append([sum(starter[0] for starter in high_score),                             sum(bench[1] for bench in high_score)])

            # Append team to valid teams
            valid_teams.append(teams[i])
            i += 1

    return valid_teams


def score_week(team, score_column):

    # Empty list for starters & bench
    start = []
    bench = []

    # Empty list for starters by position
    qb_start = []
    rb_start = []
    wr_start = []
    te_start = []
    flex_start = []

    # Empty list for bench by position
    qb_bench = []
    rb_bench = []
    wr_bench = []
    te_bench = []
    flex_bench = []

    team.sort(key=lambda x: x[score_column], reverse=True)

    # Sort players by position and drop their weekly score in the intended column
    for player in team:

        if player[3] == 'QB':
            if len(qb_start) < 1:
                qb_start.append(player[score_column])
            elif len(qb_bench) < 1:
                qb_bench.append(player[score_column])
            else:
                pass

        if player[3] == 'RB':

            if len(rb_start) < 2:
                rb_start.append(player[score_column])
            elif len(flex_start) < 1:
                flex_start.append(player[score_column])
            elif len(rb_bench) < 2:
                rb_bench.append(player[score_column])
            elif len(flex_bench) < 2:
                flex_bench.append(player[score_column])
            else:
                pass

        if player[3] == 'WR':

            if len(wr_start) < 2:
                wr_start.append(player[score_column])
            elif len(flex_start) < 1:
                flex_start.append(player[score_column])
            elif len(wr_bench) < 2:
                wr_bench.append(player[score_column])
            elif len(flex_bench) < 2:
                flex_bench.append(player[score_column])

            else:
                pass

        if player[3] == 'TE':

            if len(te_start) < 1:
                te_start.append(player[score_column])
            elif len(flex_start) < 1:
                flex_start.append(player[score_column])
            elif len(te_bench) < 1:
                te_bench.append(player[score_column])
            elif len(flex_bench) < 2:
                flex_bench.append(player[score_column])
            else:
                pass

    start = qb_start + rb_start + wr_start + te_start + flex_start
    bench = qb_bench + rb_bench + wr_bench + te_bench + flex_bench

    return [sum(start), sum(bench)]


def player_contribution(teams, players):

    # Iterate through top players
    for player in players:

        # Assign a default of max score with and without a player
        max_start_score_with = 0
        max_start_score_without = 0
        max_bench_score_with = 0
        max_bench_score_without = 0

        # Assign a default of max low score with and without a player
        max_start_low_score_with = 0
        max_start_low_score_without = 0
        max_bench_low_score_with = 0
        max_bench_low_score_without = 0

        # Assign a default of max high score with and without a player
        max_start_high_score_with = 0
        max_start_high_score_without = 0
        max_bench_high_score_with = 0
        max_bench_high_score_without = 0

        # Iterate through each team to check for a player
        for team in teams:
            with_team = False

            # Iterate through roster for each team
            for team_player in team[:-3]:

                # If player is in the team, flag as true
                if team_player[9] == player[9]:
                        with_team = True

            # If player is in team check score and record if higher than current max start score with
            if with_team is True:
                if max_start_score_with < team[-3][0]:
                    max_start_score_with = team[-3][0]
                if max_bench_score_with < team[-3][1]:
                    max_bench_score_with = team[-3][1]

                if max_start_low_score_with < team[-2][0]:
                    max_start_low_score_with = team[-2][0]
                if max_bench_low_score_with < team[-2][1]:
                    max_bench_low_score_with = team[-2][1]

                if max_start_high_score_with < team[-1][0]:
                    max_start_high_score_with = team[-1][0]
                if max_bench_high_score_with < team[-1][1]:
                    max_bench_high_score_with = team[-1][1]

            # If player is not on team, check score and record if higher than current max start score without
            else:
                if max_start_score_without < team[-3][0]:
                    max_start_score_without = team[-3][0]
                if max_bench_score_without < team[-3][1]:
                    max_bench_score_without = team[-3][1]

                if max_start_low_score_without < team[-2][0]:
                    max_start_low_score_without = team[-2][0]
                if max_bench_low_score_without < team[-2][1]:
                    max_bench_low_score_without = team[-2][1]

                if max_start_high_score_without < team[-1][0]:
                    max_start_high_score_without = team[-1][0]
                if max_bench_high_score_without < team[-1][1]:
                    max_bench_high_score_without = team[-1][1]

        # Append difference in scores to player
        player.append(max_start_score_with - max_start_score_without)
        player.append(max_bench_score_with - max_bench_score_without)

        player.append(max_start_low_score_with - max_start_low_score_without)
        player.append(max_bench_low_score_with - max_bench_low_score_without)

        player.append(max_start_high_score_with - max_start_high_score_without)
        player.append(max_bench_high_score_with - max_bench_high_score_without)

    return players


def rank_players(players, available_players, pick=80, total_picks=160):

    headers = list(available_players.columns)
    headers += ['avg_starter_spread', 'avg_bench_spread', 'lower_starter_spread',                 'lower_bench_spread', 'upper_starter_spread', 'upper_bench_spread']

    player_df = pd.DataFrame(players, columns=headers)

    # For player's with more than 50% probability of last to next pick, create risk variable.
    player_df['gamble'] = player_df['survival_probability'].apply(gamble)

    draft_status = pick / total_picks
    center_weight = 0.68
    outer_weight = 1 - center_weight
    floor_weight = outer_weight - (outer_weight * draft_status)
    ceiling_weight = outer_weight - floor_weight

    # Weight lower/mid/upper point spread
    player_df['starter_spread'] = center_weight * player_df['avg_starter_spread']                                   + floor_weight * player_df['lower_starter_spread']                                   + ceiling_weight * player_df['upper_starter_spread']

    # Rank lower/mid/upper point spread
    player_df['starter_spread_rank'] = player_df['starter_spread'].rank(ascending=0)


    # Weight lower/mid/upper point spread
    player_df['bench_spread'] = center_weight * player_df['avg_bench_spread']                                   + floor_weight * player_df['lower_bench_spread']                                   + ceiling_weight * player_df['upper_bench_spread']

    # Rank lower/mid/upper point spread
    player_df['bench_spread_rank'] = player_df['bench_spread'].rank(ascending=0)


    # Weight lower/mid/upper point spread
    starter_weight = 0.80
    bench_weight = 1 - starter_weight
    player_df['spread'] = starter_weight * player_df['starter_spread'].rank(ascending=1, pct=True)                           + bench_weight * player_df['bench_spread'].rank(ascending=1, pct=True)

    # Rank lower/mid/upper point spread
    player_df['spread_rank'] = player_df['spread'].rank(ascending=0)


    # Weight lower/mid/upper point spread
    player_df['value_over_replacement'] = center_weight * player_df['avg_value_over_replacement']                                   + floor_weight * player_df['lower_value_over_replacement']                                   + ceiling_weight * player_df['upper_value_over_replacement']

    # Rank lower/mid/upper point spread
    player_df['value_over_replacement_rank'] = player_df['value_over_replacement'].rank(ascending=0)


    # Weight lower/mid/upper point spread
    player_df['points_zscore'] = center_weight * player_df['avg_points_zscore']                                   + floor_weight * player_df['lower_points_zscore']                                   + ceiling_weight * player_df['upper_points_zscore']

    # Rank lower/mid/upper point spread
    player_df['points_zscore_rank'] = player_df['points_zscore'].rank(ascending=0)


    # Rank by average ranks
    player_df['suggestion'] = player_df['gamble']                                 * ((45 * player_df['spread'].rank(ascending=1, pct=True))                                 + (35 * player_df['value_over_replacement'].rank(ascending=1, pct=True))                                 + (20 * player_df['points_zscore'].rank(ascending=1, pct=True)))

    # Rank lower/mid/upper point spread
    player_df['suggestion_rank'] = player_df['suggestion'].rank(ascending=0)

    main_headers = ['suggestion_rank', 'player', 'survival_probability', 'playerposition', 'team', 'overallRank',                     'spread', 'spread_rank', 'value_over_replacement_rank', 'points_zscore_rank',                     'starter_spread', 'starter_spread_rank',                     'avg_starter_spread', 'lower_starter_spread', 'upper_starter_spread',                     'avg_value_over_replacement', 'lower_value_over_replacement', 'upper_value_over_replacement',                     'avg_points_zscore', 'lower_points_zscore', 'upper_points_zscore',                     'bench_spread', 'bench_spread_rank',                     'avg_bench_spread', 'lower_bench_spread', 'upper_bench_spread']

    player_df = player_df[main_headers].sort_values(by=['suggestion_rank'], ascending=True).reset_index(drop=True)

    return player_df


# For player's with more than 50% probability of last to next pick, create risk variable.
def gamble(array):

    # Level of probability before suggestion rank is affected (reduced)
    safety_threshold = 60
    # Only apply to players over threshold
    if array >= safety_threshold:
        return ((100 + safety_threshold) - array) / 100

    else:
        return 1

    return array


def show_ranks(player_df):

    print('{rank:<3s}'.format(rank='#')       + '{name:^12s}'.format(name='name')       + '{probability:^5s}'.format(probability='sur')       + '{position:^5s}'.format(position='pos')       + '{team:^4s}'.format(team='tm')       + '{value_over_replacement_rank:^3s}'.format(value_over_replacement_rank='#')       + '{avg_value_over_replacement:^5s}'.format(avg_value_over_replacement='vor')       + '{lower_value_over_replacement:^5s}'.format(lower_value_over_replacement='low')       + '{upper_value_over_replacement:^5s}'.format(upper_value_over_replacement='up')       + '{points_zscore_rank:^3s}'.format(points_zscore_rank='#')       + '{avg_points_zscore:^5s}'.format(avg_points_zscore='zsc')       + '{lower_points_zscore:^5s}'.format(lower_points_zscore='low')       + '{upper_points_zscore:^5s}'.format(upper_points_zscore='up')       + '{starter_spread_rank:^3s}'.format(starter_spread_rank='#')       + '{starter_spread:^5s}'.format(starter_spread='sts')
      + '{avg_starter_spread:^5s}'.format(avg_starter_spread='avg') \
      + '{lower_starter_spread:^5s}'.format(lower_starter_spread='low') \
      + '{upper_starter_spread:^5s}'.format(upper_starter_spread='up') \
      + '{bench_spread_rank:^3s}'.format(bench_spread_rank='#') \
      + '{bench_spread:^5s}'.format(bench_spread='bhs') \
      + '{avg_bench_spread:^5s}'.format(avg_bench_spread='avg') \
      + '{lower_bench_spread:^5s}'.format(lower_bench_spread='low') \
      + '{upper_bench_spread:^5s}'.format(upper_bench_spread='up'))

    for index, row in player_df.iterrows():
        print('{rank:<3.0f}'.format(rank=row['suggestion_rank'])               + '{name:<12s}'.format(name=row['player'][:11])               + '{probability:^5.1f}'.format(probability=row['survival_probability'])               + '{position:^5s}'.format(position=row['playerposition'])               + '{team:^4s}'.format(team=row['team'])               + '{value_over_replacement_rank:^3.0f}'.format(value_over_replacement_rank=row['value_over_replacement_rank'])               + '{avg_value_over_replacement:^5.1f}'.format(avg_value_over_replacement=row['avg_value_over_replacement'])               + '{lower_value_over_replacement:^5.0f}'.format(lower_value_over_replacement=row['lower_value_over_replacement'])               + '{upper_value_over_replacement:^5.0f}'.format(upper_value_over_replacement=row['upper_value_over_replacement'])               + '{points_zscore_rank:^3.0f}'.format(points_zscore_rank=row['points_zscore_rank'])               + '{avg_points_zscore:^5.1f}'.format(avg_points_zscore=row['avg_points_zscore'])               + '{lower_points_zscore:^5.1f}'.format(lower_points_zscore=row['lower_points_zscore'])               + '{upper_points_zscore:^5.1f}'.format(upper_points_zscore=row['upper_points_zscore'])               + '{starter_spread_rank:^3.0f}'.format(starter_spread_rank=row['starter_spread_rank'])               + '{starter_spread:^5.1f}'.format(starter_spread=row['starter_spread'])
              + '{avg_starter_spread:^5.1f}'.format(avg_starter_spread=row['avg_starter_spread']) \
              + '{lower_starter_spread:^5.1f}'.format(lower_starter_spread=row['lower_starter_spread']) \
              + '{upper_starter_spread:^5.1f}'.format(upper_starter_spread=row['upper_starter_spread']) \
              + '{bench_spread_rank:^3.0f}'.format(bench_spread_rank=row['bench_spread_rank']) \
              + '{bench_spread:^5.1f}'.format(bench_spread=row['bench_spread']) \
              + '{avg_bench_spread:^5.1f}'.format(avg_bench_spread=row['avg_bench_spread']) \
              + '{lower_bench_spread:^5.1f}'.format(lower_bench_spread=row['lower_bench_spread']) \
              + '{upper_bench_spread:^5.1f}'.format(upper_bench_spread=row['upper_bench_spread']))


def show_players(df):

    view = df[['player', 'playerposition', 'team', 'ADP', 'points', 'positionRank']]

    print('\n')
    print('{index:^15s}'.format(index='Index')           + '{player:15s}'.format(player='Player')           + '{playerposition:^15s}'.format(playerposition='Position')           + '{team:^15s}'.format(team='Team')           + '{ADP:^15s}'.format(ADP='ADP')           + '{points:^15s}'.format(points='Proj. Points')           + '{positionRank:^15s}'.format(positionRank='Position Rank'))

    for index, row in df.iterrows():

        print('{index:^15d}'.format(index=index)               + '{player:15s}'.format(player=row['player'][:15])               + '{playerposition:^15s}'.format(playerposition=row['playerposition'][:15])               + '{team:^15s}'.format(team=row['team'])               + '{ADP:^15.1f}'.format(ADP=row['ADP'])               + '{points:^15.1f}'.format(points=row['points'])               + '{positionRank:^15.0f}'.format(positionRank=row['positionRank']))

    print('\n')


def player_search(df, verbiage):

    search = df.copy()
    search = search.sort_values(by=['ADP'], ascending=True)
    show_players(search[:10])

    valid = False
    while valid is False:
        index = input("\n" + verbiage + "\n")

        try:
            # Check if error results for changing to integer type
            index = int(index)
            try:
                return df['playerId'][index]
            except:
                continue

        except:

            if index.lower() == 'skip':
                print("\nSkipping.")
                return None

            elif index.lower() == 'roster remove':
                return 'roster remove'

            elif index.lower() == 'roster add':
                return 'roster add'

            elif index.lower() == 'player remove':
                return 'player remove'

            elif index.lower() == 'player add':
                return 'player add'

            elif index.lower() == 'roster':
                return 'roster'

            elif index.lower() == 'recalculate':
                return 'recalculate'

            elif type(index) is str:
                show_players(search.ix[(search['player'].str.contains(index, case=False)) |                                        (search['playerposition'].str.contains(index, case=False)) |                                        (search['team'].str.contains(index, case=False))][:5])

            else:
                continue


def picks(rounds, teams, pick):

    picks = []

    # Append picks for each round in a snake draft for user's position
    for round in range(1, rounds + 1):

        if round % 2 != 0:
            picks.append(((round - 1) * teams) + pick)
        else:
            picks.append((round * teams) - pick + 1)

    picks.append(1)

    return picks


def draft_assistant(rounds, league_teams, user_pick):

    # Define user draft positions
    user_picks = picks(rounds, league_teams, user_pick)

    # Read in data, clean, and merge source datasets
    available = player_data()

    # Add features to dataset
    backup = add_features(df=available, pick=1, next_pick=user_picks[1])
    available = add_features(df=available, pick=1, next_pick=user_picks[1])

    # Create seperate dataframes for drafted and rostered players
    roster = pd.DataFrame(columns=available.columns)
    drafted = pd.DataFrame(columns=available.columns)

    # State player selection in advance to avoid clutter
    # State player selection in advance to avoid clutter
    print("Thanks for using the draft assistant! A prompt will pop up after each pick where you are asked to\n"          + "select a player drafted. By default the top ten most likely players are returned but if another\n"          + "player was selected search by their index, name, team, or position.\n\n"          + "'skip' - skip turn\n"          + "'roster remove' - remove player from your roster\n"          + "'roster add' - add player to your roster\n"          + "'player remove' - remove player from available players\n"          + "'player add' - add player to available players\n")

    # Iterate through all picks
    for pick in range(1, (rounds * league_teams) + 1):

        draft_round = int(np.ceil(pick/league_teams))

        print("\nRound - " + str(draft_round) + ", Pick - " + str(pick))

        if pick == user_picks[0]:

            # Add features to dataset
            available = add_features(df=available, pick=pick, next_pick=user_picks[1])
            # Grab top players by position
            players = top_players(df=available, roster=roster)
            print("\nTesting team combinations, please wait...\n")
            # Create and validate potential teams
            teams = make_teams(players=players, roster=roster)
            # Grab top players and find contribution
            players = player_contribution(teams, players)
            # Add defense to top players
            players += grab_special(df=available, roster=roster)
            # Rank variables across available top players and average ranks
            ranks = rank_players(players = players,                                  available_players = available,                                  pick=pick,                                  total_picks=(rounds * league_teams))

            show_ranks(ranks)

            # Loop condition
            player_picked = False
            while player_picked is False:

                # Select player to add to roster
                player_id = player_search(df=available, verbiage="**YOUR ROSTER**")

                if player_id == 'roster remove':
                    player_id = player_search(df=roster.sort_values(by=['playerposition'], ascending=True), verbiage="**REMOVE PLAYER FROM ROSTER**")
                    roster = roster[roster['playerId'] != player_id]

                elif player_id == 'roster add':
                    player_id = player_search(df=backup, verbiage="**ADD PLAYER TO ROSTER**")
                    roster = roster.append(backup[backup['playerId'] == player_id])

                elif player_id == 'player remove':
                    player_id = player_search(df=available, verbiage="**REMOVE PLAYER FROM AVAILABLE PLAYERS**")
                    available = available[available['playerId'] != player_id]

                elif player_id == 'player add':
                    player_id = player_search(df=backup, verbiage="**ADD PLAYER TO AVAILABLE PLAYERS**")
                    available = available.append(backup[backup['playerId'] == player_id])

                elif player_id == 'roster':
                    show_players(roster)

                elif player_id == 'recalculate':
                    # Add features to dataset
                    available = add_features(df=available, pick=pick, next_pick=user_picks[1])
                    # Grab top players by position
                    players = top_players(df=available, roster=roster)
                    print("\nTesting team combinations, please wait...\n")
                    # Create and validate potential teams
                    teams = make_teams(players=players, roster=roster)
                    # Grab top players and find contribution
                    players = player_contribution(teams, players)
                    # Add defense to top players
                    players += grab_special(df=available, roster=roster)
                    # Rank variables across available top players and average ranks
                    ranks = rank_players(players = players,                                          available_players = available,                                          pick=pick,                                          total_picks=(rounds * league_teams))

                    show_ranks(ranks)

                else:
                    player_picked = True


            # Append player to roster dataframe
            roster = roster.append(available[available['playerId'] == player_id])
            # Append player to drafted dataframe
            drafted = drafted.append(available[available['playerId'] == player_id])
            # Remove player from available dataframe
            available = available[available['playerId'] != player_id]
            # Remove pick from user picks
            user_picks.pop(0)

        else:

            # Loop condition
            player_picked = False
            while player_picked is False:

                # Select player to add to roster
                player_id = player_search(df=available, verbiage="**OPPONENT PICK**")

                if player_id == 'roster remove':
                    player_id = player_search(df=roster.sort_values(by=['playerposition'], ascending=True), verbiage="**REMOVE PLAYER FROM ROSTER**")
                    roster = roster[roster['playerId'] != player_id]

                elif player_id == 'roster add':
                    player_id = player_search(df=backup, verbiage="**ADD PLAYER TO ROSTER**")
                    roster = roster.append(backup[backup['playerId'] == player_id])

                elif player_id == 'player remove':
                    player_id = player_search(df=available, verbiage="**REMOVE PLAYER FROM AVAILABLE PLAYERS**")
                    available = available[available['playerId'] != player_id]

                elif player_id == 'player add':
                    player_id = player_search(df=backup, verbiage="**ADD PLAYER TO AVAILABLE PLAYERS**")
                    available = available.append(backup[backup['playerId'] == player_id])

                elif player_id == 'roster':
                    show_players(roster)

                elif player_id == 'recalculate':
                    # Add features to dataset
                    available = add_features(df=available, pick=pick, next_pick=user_picks[1])
                    # Grab top players by position
                    players = top_players(df=available, roster=roster)
                    print("\nTesting team combinations, please wait...\n")
                    # Create and validate potential teams
                    teams = make_teams(players=players, roster=roster)
                    # Grab top players and find contribution
                    players = player_contribution(teams, players)
                    # Add defense to top players
                    players += grab_special(df=available, roster=roster)
                    # Rank variables across available top players and average ranks
                    ranks = rank_players(players = players,                                          available_players = available,                                          pick=pick,                                          total_picks=(rounds * league_teams))

                    show_ranks(ranks)

                else:
                    player_picked = True

            # Append player to drafted dataframe
            drafted = drafted.append(available[available['playerId'] == player_id])
            # Remove player from available dataframe
            available = available[available['playerId'] != player_id]

    show_players(roster)

league_rounds = int(input("How many rounds? "))
league_teams = int(input("How many teams? "))
league_pick = int(input("What is your first pick? "))

draft_assistant(league_rounds, league_teams, league_pick)
