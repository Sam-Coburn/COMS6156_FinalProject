import java.io.Reader;
import java.io.BufferedReader;
import java.io.InputStreamReader;

// 
// Decompiled by Procyon v0.5.29
// 

class Kaprekar_Transformation
{
    private static int a() {
        int int1;
        try {
            int1 = Integer.parseInt(new BufferedReader(new InputStreamReader(System.in)).readLine());
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return 0;
        }
        return int1;
    }
    
    public static void main(final String[] array) {
        System.out.print("Enter a 3 digit number for Kaprekar Tranformation: ");
        int a = a();
        Label_0037: {
            if (a >= 0) {
                if (a <= 999) {
                    break Label_0037;
                }
            }
            System.out.println("Your input is not correct.");
        }
        final int n = a / 100;
        final int n2 = (a - n * 100) / 10;
        final int n3 = a % 10;
        if (n == n2 && n == n3) {
            System.out.println("All digits should not be equal.");
            return;
        }
        while (true) {
            int n4 = a / 100;
            final int n5 = (a - n4 * 100) / 10;
            final int n6 = a % 10;
            int n7;
            int n8;
            int n9;
            if (n4 <= n5) {
                if (n5 <= n6) {
                    n7 = n6;
                    n8 = n5;
                    n9 = n4;
                }
                else if (n4 <= n6) {
                    n7 = n5;
                    n8 = n6;
                    n9 = n4;
                }
                else {
                    n7 = n5;
                    n8 = n4;
                    n9 = n6;
                }
            }
            else if (n4 <= n6) {
                n7 = n6;
                n8 = n4;
                n9 = n5;
            }
            else if (n5 <= n6) {
                n7 = n4;
                n8 = n6;
                n9 = n5;
            }
            else {
                n7 = n4;
                n8 = n5;
                n9 = n6;
            }
            int n10;
            int n11;
            if (n4 >= n5) {
                if (n5 >= n6) {
                    n10 = n6;
                    n11 = n5;
                }
                else if (n4 >= n6) {
                    n10 = n5;
                    n11 = n6;
                }
                else {
                    n10 = n5;
                    n11 = n4;
                    n4 = n6;
                }
            }
            else if (n4 >= n6) {
                n10 = n6;
                n11 = n4;
                n4 = n5;
            }
            else if (n5 >= n6) {
                n10 = n4;
                n11 = n6;
                n4 = n5;
            }
            else {
                n10 = n4;
                n11 = n5;
                n4 = n6;
            }
            final int i = n7 * 100 + n8 * 10 + n9;
            final int j = n10 * 100 + n11 * 10 + n4;
            final int k = i - j;
            System.out.format("%d - %d = %d\n", i, j, k);
            if (a == k) {
                break;
            }
            a = k;
        }
    }
}
