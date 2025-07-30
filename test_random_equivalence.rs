use lightgbm_rust::core::utils::random::Random;

fn main() {
    println!("Rust Random Implementation Test");
    println!("===============================");
    
    // Test with same seed as C++ version
    let mut rng = Random::with_seed(42);
    
    println!("Seed: 42");
    
    // Test NextShort
    print!("NextShort(0, 100) x5: ");
    for _ in 0..5 {
        print!("{} ", rng.next_short(0, 100));
    }
    println!();

    // Reset and test NextInt
    let mut rng2 = Random::with_seed(42);
    print!("NextInt(0, 1000) x5: ");
    for _ in 0..5 {
        print!("{} ", rng2.next_int(0, 1000));
    }
    println!();

    // Reset and test NextFloat
    let mut rng3 = Random::with_seed(42);
    print!("NextFloat() x5: ");
    for _ in 0..5 {
        print!("{:.6} ", rng3.next_float());
    }
    println!();

    // Test Sample
    let mut rng4 = Random::with_seed(42);
    print!("Sample(10, 3): ");
    let result = rng4.sample(10, 3);
    for val in &result {
        print!("{} ", val);
    }
    println!("(size={})", result.len());
    
    println!("\nExpected C++ output:");
    println!("NextShort(0, 100) x5: 75 0 69 56 83");
    println!("NextInt(0, 1000) x5: 557 348 367 998 289");
    println!("NextFloat() x5: 0.005341 0.012207 0.545319 0.917236 0.490814");
    println!("Sample(10, 3): 5 7 8 (size=3)");
}