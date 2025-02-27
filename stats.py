from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from typing import List, Dict, Optional, Union
import uvicorn
from pydantic import BaseModel
from datetime import datetime
import json
import os
import pandas as pd
import fastf1
import logging
from fastf1.core import Session
from pathlib import Path

# Set up logging to terminal
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("f1-stats-api")

# Configure FastF1 cache
cache_dir = Path("./fastf1_cache")
if not cache_dir.exists():
    cache_dir.mkdir(parents=True)
fastf1.Cache.enable_cache(str(cache_dir))

app = FastAPI(title="F1 Statistics API", description="API for Formula 1 statistics and data")

# Data models
class Driver(BaseModel):
    id: str
    name: str
    team: str
    nationality: str
    number: int
    championships: int
    career_points: float
    career_wins: int
    career_podiums: int

class Team(BaseModel):
    id: str
    name: str
    nationality: str
    championships: int
    first_season: int
    base: str
    current_drivers: List[str]

class Race(BaseModel):
    id: str
    name: str
    circuit: str
    date: datetime
    location: str
    country: str
    number_of_laps: int
    circuit_length_km: float

class RaceResult(BaseModel):
    race_id: str
    driver_id: str
    team_id: str
    position: int
    points: float
    fastest_lap: bool = False
    dnf: bool = False
    dnf_reason: Optional[str] = None

class LapTime(BaseModel):
    driver_id: str
    lap_number: int
    time_seconds: float
    position: int
    compound: Optional[str] = None
    sector_1_time: Optional[float] = None
    sector_2_time: Optional[float] = None
    sector_3_time: Optional[float] = None

class SessionData(BaseModel):
    session_name: str
    session_date: datetime
    session_type: str
    circuit_name: str
    available_data: List[str]

# Sample data (in a real application, this would come from a database)
drivers = [
    Driver(
        id="VER",
        name="Max Verstappen",
        team="Red Bull Racing",
        nationality="Dutch",
        number=1,
        championships=3,
        career_points=2511.5,
        career_wins=56,
        career_podiums=98
    ),
    Driver(
        id="HAM",
        name="Lewis Hamilton",
        team="Mercedes",
        nationality="British",
        number=44,
        championships=7,
        career_points=4639.5,
        career_wins=103,
        career_podiums=199
    ),
    Driver(
        id="LEC",
        name="Charles Leclerc",
        team="Ferrari",
        nationality="Monegasque",
        number=16,
        championships=0,
        career_points=1002.0,
        career_wins=5,
        career_podiums=28
    ),
]

teams = [
    Team(
        id="RBR",
        name="Red Bull Racing",
        nationality="Austrian",
        championships=6,
        first_season=2005,
        base="Milton Keynes, United Kingdom",
        current_drivers=["VER", "PER"]
    ),
    Team(
        id="MER",
        name="Mercedes",
        nationality="German",
        championships=8,
        first_season=2010,
        base="Brackley, United Kingdom",
        current_drivers=["HAM", "RUS"]
    ),
    Team(
        id="FER",
        name="Ferrari",
        nationality="Italian",
        championships=16,
        first_season=1950,
        base="Maranello, Italy",
        current_drivers=["LEC", "SAI"]
    ),
]

races = [
    Race(
        id="BHR2024",
        name="Bahrain Grand Prix",
        circuit="Bahrain International Circuit",
        date=datetime(2024, 3, 2),
        location="Sakhir",
        country="Bahrain",
        number_of_laps=57,
        circuit_length_km=5.412
    ),
    Race(
        id="SAU2024",
        name="Saudi Arabian Grand Prix",
        circuit="Jeddah Corniche Circuit",
        date=datetime(2024, 3, 9),
        location="Jeddah",
        country="Saudi Arabia",
        number_of_laps=50,
        circuit_length_km=6.174
    ),
]

race_results = [
    RaceResult(
        race_id="BHR2024",
        driver_id="VER",
        team_id="RBR",
        position=1,
        points=25.0,
        fastest_lap=True
    ),
    RaceResult(
        race_id="BHR2024",
        driver_id="HAM",
        team_id="MER",
        position=5,
        points=10.0
    ),
    RaceResult(
        race_id="BHR2024",
        driver_id="LEC",
        team_id="FER",
        position=4,
        points=12.0
    ),
    RaceResult(
        race_id="SAU2024",
        driver_id="VER",
        team_id="RBR",
        position=1,
        points=25.0
    ),
    RaceResult(
        race_id="SAU2024",
        driver_id="HAM",
        team_id="MER",
        position=9,
        points=2.0
    ),
    RaceResult(
        race_id="SAU2024",
        driver_id="LEC",
        team_id="FER",
        position=3,
        points=15.0,
        fastest_lap=True
    ),
]

# Custom JSON encoder to handle datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        return super().default(obj)

def pretty_print(data):
    """Convert Pydantic models to dict and print as formatted JSON"""
    if isinstance(data, list):
        data = [item.dict() if hasattr(item, 'dict') else item for item in data]
    elif hasattr(data, 'dict'):
        data = data.dict()
    
    formatted_json = json.dumps(data, indent=2, cls=CustomJSONEncoder)
    logger.info(f"\n{formatted_json}")
    return data

# Helper functions for FastF1 data
def load_session(year: int, gp: str, session_type: str = "R") -> Session:
    """Load a particular F1 session from FastF1"""
    logger.info(f"Loading F1 session data: {year} {gp} {session_type}")
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load()
        return session
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        raise HTTPException(status_code=404, detail=f"Session data not found: {year} {gp} {session_type}")

def fetch_available_sessions(year: int) -> List[Dict]:
    """Fetch all available sessions for a given year"""
    try:
        schedule = fastf1.get_event_schedule(year)
        events = []
        for idx, event in schedule.iterrows():
            events.append({
                "round": int(event["RoundNumber"]),
                "name": event["EventName"],
                "country": event["Country"],
                "date": event["EventDate"],
                "sessions": {
                    "fp1": event["Session1"],
                    "fp2": event["Session2"],
                    "fp3": event.get("Session3", None),
                    "qualifying": event.get("Session4", None),
                    "sprint": event.get("Sprint", None),
                    "race": event["Session5"]
                }
            })
        return events
    except Exception as e:
        logger.error(f"Error fetching available sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching available sessions: {str(e)}")

def convert_lap_data(lap_data) -> List[LapTime]:
    """Convert FastF1 lap data to our API format"""
    result = []
    for _, lap in lap_data.iterrows():
        try:
            # Convert timedelta to seconds
            lap_time_seconds = lap.get('LapTime', pd.Timedelta(0)).total_seconds() if not pd.isna(lap.get('LapTime')) else None
            s1_time = lap.get('Sector1Time', pd.Timedelta(0)).total_seconds() if not pd.isna(lap.get('Sector1Time')) else None
            s2_time = lap.get('Sector2Time', pd.Timedelta(0)).total_seconds() if not pd.isna(lap.get('Sector2Time')) else None
            s3_time = lap.get('Sector3Time', pd.Timedelta(0)).total_seconds() if not pd.isna(lap.get('Sector3Time')) else None
            
            # Skip invalid laps
            if lap_time_seconds is None or lap_time_seconds <= 0:
                continue
                
            result.append(LapTime(
                driver_id=lap.get('Driver', ''),
                lap_number=int(lap.get('LapNumber', 0)),
                time_seconds=lap_time_seconds,
                position=int(lap.get('Position', 0)) if not pd.isna(lap.get('Position')) else 0,
                compound=lap.get('Compound', None),
                sector_1_time=s1_time,
                sector_2_time=s2_time,
                sector_3_time=s3_time
            ))
        except Exception as e:
            logger.warning(f"Error converting lap: {e}")
            continue
    return result

def fetch_session_telemetry(session, driver_id: str):
    """Fetch detailed telemetry for a specific driver in a session"""
    try:
        driver_data = session.laps.pick_driver(driver_id)
        if driver_data.empty:
            return {"error": f"No data found for driver {driver_id}"}
        
        # Get fastest lap for telemetry
        fastest_lap = driver_data.pick_fastest()
        
        # Get telemetry data
        telemetry = fastest_lap.get_telemetry()
        
        # Convert to dict for JSON
        telemetry_dict = {
            "driver": driver_id,
            "lap_number": int(fastest_lap['LapNumber']),
            "lap_time": fastest_lap['LapTime'].total_seconds(),
            "data_points": len(telemetry),
            "speed": telemetry['Speed'].tolist(),
            "throttle": telemetry['Throttle'].tolist(),
            "brake": telemetry['Brake'].tolist(),
            "gear": telemetry['nGear'].tolist(),
            "rpm": telemetry['RPM'].tolist(),
            "distance": telemetry['Distance'].tolist()
        }
        
        return telemetry_dict
    except Exception as e:
        logger.error(f"Error fetching telemetry: {e}")
        return {"error": str(e)}

# Background tasks
def download_session_data(year: int, gp: str, session_type: str = "R"):
    """Background task to download and cache F1 session data"""
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load()
        logger.info(f"Successfully downloaded and cached data for {year} {gp} {session_type}")
    except Exception as e:
        logger.error(f"Error downloading session data: {e}")

# Middleware to log all requests and responses
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request path: {request.url.path}")
    response = await call_next(request)
    return response

# API routes
@app.get("/")
def root():
    welcome_message = {
        "message": "Welcome to the F1 Statistics API", 
        "endpoints": [
            "/drivers", 
            "/teams", 
            "/races", 
            "/results",
            "/fastf1/sessions/{year}",
            "/fastf1/lap-times/{year}/{gp}/{session_type}",
            "/fastf1/telemetry/{year}/{gp}/{session_type}/{driver_id}",
            "/stats/driver-standings",
            "/stats/team-standings"
        ]
    }
    logger.info("\n=== F1 STATISTICS API ===")
    pretty_print(welcome_message)
    return welcome_message

@app.get("/drivers", response_model=List[Driver])
def get_drivers(nationality: Optional[str] = None):
    logger.info("\n=== DRIVERS ===")
    result = [d for d in drivers if not nationality or d.nationality.lower() == nationality.lower()]
    return pretty_print(result)

@app.get("/drivers/{driver_id}", response_model=Driver)
def get_driver(driver_id: str):
    logger.info(f"\n=== DRIVER: {driver_id} ===")
    for driver in drivers:
        if driver.id == driver_id:
            return pretty_print(driver)
    raise HTTPException(status_code=404, detail="Driver not found")

@app.get("/teams", response_model=List[Team])
def get_teams():
    logger.info("\n=== TEAMS ===")
    return pretty_print(teams)

@app.get("/teams/{team_id}", response_model=Team)
def get_team(team_id: str):
    logger.info(f"\n=== TEAM: {team_id} ===")
    for team in teams:
        if team.id == team_id:
            return pretty_print(team)
    raise HTTPException(status_code=404, detail="Team not found")

@app.get("/races", response_model=List[Race])
def get_races(country: Optional[str] = None):
    logger.info("\n=== RACES ===")
    result = [r for r in races if not country or r.country.lower() == country.lower()]
    return pretty_print(result)

@app.get("/races/{race_id}", response_model=Race)
def get_race(race_id: str):
    logger.info(f"\n=== RACE: {race_id} ===")
    for race in races:
        if race.id == race_id:
            return pretty_print(race)
    raise HTTPException(status_code=404, detail="Race not found")

@app.get("/results", response_model=List[RaceResult])
def get_results(race_id: Optional[str] = None, driver_id: Optional[str] = None):
    logger.info("\n=== RACE RESULTS ===")
    filtered_results = race_results
    
    if race_id:
        filtered_results = [r for r in filtered_results if r.race_id == race_id]
    
    if driver_id:
        filtered_results = [r for r in filtered_results if r.driver_id == driver_id]
    
    return pretty_print(filtered_results)

@app.get("/stats/driver-standings")
def get_driver_standings():
    logger.info("\n=== DRIVER STANDINGS ===")
    standings = {}
    
    for result in race_results:
        driver_id = result.driver_id
        if driver_id not in standings:
            standings[driver_id] = {"points": 0, "wins": 0, "podiums": 0}
        
        standings[driver_id]["points"] += result.points
        
        if result.position == 1:
            standings[driver_id]["wins"] += 1
        
        if result.position <= 3:
            standings[driver_id]["podiums"] += 1
    
    # Enrich with driver names
    enriched_standings = []
    for driver_id, stats in standings.items():
        driver = next((d for d in drivers if d.id == driver_id), None)
        if driver:
            enriched_standings.append({
                "driver_id": driver_id,
                "name": driver.name,
                "team": driver.team,
                "points": stats["points"],
                "wins": stats["wins"],
                "podiums": stats["podiums"]
            })
    
    # Sort by points (descending)
    result = sorted(enriched_standings, key=lambda x: x["points"], reverse=True)
    return pretty_print(result)

@app.get("/stats/team-standings")
def get_team_standings():
    logger.info("\n=== TEAM STANDINGS ===")
    standings = {}
    
    for result in race_results:
        team_id = result.team_id
        if team_id not in standings:
            standings[team_id] = {"points": 0, "wins": 0, "podiums": 0}
        
        standings[team_id]["points"] += result.points
        
        if result.position == 1:
            standings[team_id]["wins"] += 1
        
        if result.position <= 3:
            standings[team_id]["podiums"] += 1
    
    # Enrich with team names
    enriched_standings = []
    for team_id, stats in standings.items():
        team = next((t for t in teams if t.id == team_id), None)
        if team:
            enriched_standings.append({
                "team_id": team_id,
                "name": team.name,
                "nationality": team.nationality,
                "points": stats["points"],
                "wins": stats["wins"],
                "podiums": stats["podiums"]
            })
    
    # Sort by points (descending)
    result = sorted(enriched_standings, key=lambda x: x["points"], reverse=True)
    return pretty_print(result)

# FastF1 API endpoints
@app.get("/fastf1/sessions/{year}")
def get_available_sessions(year: int):
    """Get all available race weekends for a given year"""
    logger.info(f"\n=== AVAILABLE SESSIONS FOR {year} ===")
    try:
        events = fetch_available_sessions(year)
        return pretty_print(events)
    except Exception as e:
        logger.error(f"Error fetching available sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastf1/lap-times/{year}/{gp}/{session_type}")
def get_lap_times(
    year: int, 
    gp: str, 
    session_type: str = Path("R", description="Session type: FP1, FP2, FP3, Q, S, R (for Race)"),
    driver_id: Optional[str] = Query(None, description="Filter by driver code (e.g., VER, HAM, LEC)")
):
    """Get lap times from a specific session"""
    logger.info(f"\n=== LAP TIMES FOR {year} {gp} {session_type} ===")
    try:
        session = load_session(year, gp, session_type)
        
        # Filter by driver if specified
        laps = session.laps
        if driver_id:
            laps = laps.pick_driver(driver_id)
            if laps.empty:
                raise HTTPException(status_code=404, detail=f"No data found for driver {driver_id}")
        
        # Convert to our API format
        lap_times = convert_lap_data(laps)
        
        return pretty_print({
            "event": gp,
            "year": year,
            "session_type": session_type,
            "lap_count": len(lap_times),
            "laps": [lap.dict() for lap in lap_times]
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lap times: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastf1/telemetry/{year}/{gp}/{session_type}/{driver_id}")
def get_telemetry(
    year: int, 
    gp: str, 
    session_type: str = Query("R", description="Session type: FP1, FP2, FP3, Q, S, R (for Race)"),
    driver_id: str = Query(..., description="Driver code (e.g., VER, HAM, LEC)")
):
    """Get detailed telemetry for a driver's fastest lap"""
    logger.info(f"\n=== TELEMETRY FOR {driver_id} IN {year} {gp} {session_type} ===")
    try:
        session = load_session(year, gp, session_type)
        telemetry = fetch_session_telemetry(session, driver_id)
        return pretty_print(telemetry)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastf1/cache/{year}/{gp}")
def cache_race_weekend(
    year: int, 
    gp: str,
    background_tasks: BackgroundTasks
):
    """Trigger background download and caching of all sessions for a race weekend"""
    logger.info(f"\n=== CACHING RACE WEEKEND: {year} {gp} ===")
    
    # Queue background tasks for different session types
    session_types = ["FP1", "FP2", "FP3", "Q", "R"]
    for session_type in session_types:
        background_tasks.add_task(download_session_data, year, gp, session_type)
    
    return {"status": "success", "message": f"Background download started for {year} {gp} sessions"}

@app.get("/fastf1/season-results/{year}")
def get_season_results(year: int):
    """Get results for all races in a season"""
    logger.info(f"\n=== SEASON RESULTS FOR {year} ===")
    try:
        # Get the schedule for the year
        events = fetch_available_sessions(year)
        
        # Collect results for all completed races
        season_results = []
        for event in events:
            try:
                # Skip if race hasn't happened yet
                event_date = event["sessions"]["race"].date() if hasattr(event["sessions"]["race"], "date") else event["sessions"]["race"]
                if event_date > datetime.now().date():
                    continue
                
                # Load the race session
                session = load_session(year, event["name"], "R")
                
                # Get final classification
                results = session.results
                
                # Format the results
                race_result = {
                    "round": event["round"],
                    "name": event["name"],
                    "date": event["date"],
                    "results": []
                }
                
                for _, driver in results.iterrows():
                    race_result["results"].append({
                        "position": int(driver["Position"]) if not pd.isna(driver["Position"]) else None,
                        "driver_number": int(driver["DriverNumber"]) if not pd.isna(driver["DriverNumber"]) else None,
                        "driver_id": driver["Abbreviation"],
                        "driver_name": f"{driver['FirstName']} {driver['LastName']}",
                        "team": driver["TeamName"],
                        "points": float(driver["Points"]) if not pd.isna(driver["Points"]) else 0.0,
                        "status": driver["Status"],
                        "dnf": driver["Status"] != "Finished" and "Lap" not in driver["Status"],
                        "laps": int(driver["Laps"]),
                        "grid": int(driver["GridPosition"]) if not pd.isna(driver["GridPosition"]) else None,
                        "fastest_lap": driver["Abbreviation"] == session.laps.pick_fastest()["Driver"]
                    })
                
                season_results.append(race_result)
            except Exception as e:
                logger.warning(f"Error getting results for {event['name']}: {e}")
                continue
        
        return pretty_print({
            "year": year,
            "races": len(season_results),
            "results": season_results
        })
    except Exception as e:
        logger.error(f"Error getting season results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a CLI interface
def print_data_in_terminal():
    """Print all F1 stats data to the terminal without starting the server"""
    print("\n===== F1 STATISTICS CLI MODE =====")
    
    print("\n===== DRIVERS =====")
    for driver in drivers:
        print(f"- {driver.name} (#{driver.number}): {driver.team}, {driver.championships} championships")
    
    print("\n===== TEAMS =====")
    for team in teams:
        print(f"- {team.name} ({team.nationality}): {team.championships} championships")
    
    print("\n===== RACES =====")
    for race in races:
        print(f"- {race.name} ({race.date.strftime('%Y-%m-%d')}): {race.circuit}, {race.country}")
    
    print("\n===== DRIVER STANDINGS =====")
    driver_standings = get_driver_standings()
    for pos, driver in enumerate(driver_standings, 1):
        print(f"{pos}. {driver['name']}: {driver['points']} points, {driver['wins']} wins")
    
    print("\n===== TEAM STANDINGS =====")
    team_standings = get_team_standings()
    for pos, team in enumerate(team_standings, 1):
        print(f"{pos}. {team['name']}: {team['points']} points, {team['wins']} wins")
    
    print("\n===== RACE RESULTS =====")
    for race in races:
        print(f"\n{race.name} Results:")
        race_results_filtered = [r for r in race_results if r.race_id == race.id]
        sorted_results = sorted(race_results_filtered, key=lambda x: x.position)
        
        for result in sorted_results:
            driver_name = next((d.name for d in drivers if d.id == result.driver_id), "Unknown")
            team_name = next((t.name for t in teams if t.id == result.team_id), "Unknown")
            fastest_lap = " (FL)" if result.fastest_lap else ""
            print(f"{result.position}. {driver_name} ({team_name}): {result.points} points{fastest_lap}")

    print("\n===== FASTF1 INFORMATION =====")
    print("FastF1 is now integrated! You can access real F1 data via the API endpoints.")
    print("Try visiting these endpoints in your browser after starting the server:")
    print("- http://localhost:8000/fastf1/sessions/2023 - List all sessions from 2023")
    print("- http://localhost:8000/fastf1/lap-times/2023/Bahrain/R - Get lap times from 2023 Bahrain GP")
    print("- http://localhost:8000/fastf1/telemetry/2023/Bahrain/R/VER - Get Verstappen's telemetry from 2023 Bahrain GP")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run in CLI mode only
        print_data_in_terminal()
    else:
        # Print data first, then start the server
        print_data_in_terminal()
        print("\n===== STARTING WEB SERVER =====")
        print("Visit http://localhost:8000/docs to access the API documentation")
        uvicorn.run(app, host="0.0.0.0", port=8000)