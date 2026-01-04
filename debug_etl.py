# Save as /d/agrogram-afie/debug_etl.py
import sys
import os
import asyncio

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath('.'))

async def debug_etl():
    print("Debugging ETL Pipeline...")
    
    try:
        from src.data.pipeline.etl_pipeline import ETLPipeline
        
        pipeline = ETLPipeline()
        print("✅ Pipeline initialized")
        
        # Test extraction
        print("\nTesting extraction...")
        extract_result = await pipeline.extract_data("weather_agromonitoring")
        
        print(f"Type of extract_result: {type(extract_result)}")
        print(f"Keys in extract_result: {list(extract_result.keys()) if isinstance(extract_result, dict) else 'Not a dict'}")
        
        if isinstance(extract_result, dict):
            for key, value in extract_result.items():
                print(f"\n{key}:")
                if isinstance(value, dict):
                    print(f"  Dict keys: {list(value.keys())}")
                else:
                    print(f"  Type: {type(value)}, Value sample: {str(value)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_etl())