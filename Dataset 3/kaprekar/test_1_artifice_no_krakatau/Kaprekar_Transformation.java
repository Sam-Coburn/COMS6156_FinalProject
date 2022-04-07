import java.io.Reader;
import java.io.BufferedReader;
import java.io.InputStreamReader;

// 
// Decompiled by Procyon v0.5.29
// 

class Kaprekar_Transformation
{
    public static int m01() {
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
    
    static int m11(final int n) {
        int n2 = n / 100;
        final int n3 = (n - n2 * 100) / 10;
        final int n4 = n % 10;
        int n5;
        int n6;
        int n7;
        if (n2 <= n3) {
            if (n3 <= n4) {
                n5 = n4;
                n6 = n3;
                n7 = n2;
            }
            else if (n2 <= n4) {
                n5 = n3;
                n6 = n4;
                n7 = n2;
            }
            else {
                n5 = n3;
                n6 = n2;
                n7 = n4;
            }
        }
        else if (n2 <= n4) {
            n5 = n4;
            n6 = n2;
            n7 = n3;
        }
        else if (n3 <= n4) {
            n5 = n2;
            n6 = n4;
            n7 = n3;
        }
        else {
            n5 = n2;
            n6 = n3;
            n7 = n4;
        }
        int n8;
        int n9;
        if (n2 >= n3) {
            if (n3 >= n4) {
                n8 = n4;
                n9 = n3;
            }
            else if (n2 >= n4) {
                n8 = n3;
                n9 = n4;
            }
            else {
                n8 = n3;
                n9 = n2;
                n2 = n4;
            }
        }
        else if (n2 >= n4) {
            n8 = n4;
            n9 = n2;
            n2 = n3;
        }
        else if (n3 >= n4) {
            n8 = n2;
            n9 = n4;
            n2 = n3;
        }
        else {
            n8 = n2;
            n9 = n3;
            n2 = n4;
        }
        final int i = n5 * 100 + n6 * 10 + n7;
        final int j = n8 * 100 + n9 * 10 + n2;
        final int k = i - j;
        System.out.format("%d - %d = %d\n", i, j, k);
        return k;
    }
    
    public static void main(final String[] array) {
        System.out.print("Enter a 3 digit number for Kaprekar Tranformation: ");
        int m01 = m01();
        Label_0037: {
            if (m01 >= 0) {
                if (m01 <= 999) {
                    break Label_0037;
                }
            }
            System.out.println("Your input is not correct.");
        }
        final int n = m01 / 100;
        final int n2 = (m01 - n * 100) / 10;
        final int n3 = m01 % 10;
        if (n == n2 && n == n3) {
            System.out.println("All digits should not be equal.");
            return;
        }
        while (true) {
            final int m2 = m11(m01);
            if (m01 == m2) {
                break;
            }
            m01 = m2;
        }
    }
}
