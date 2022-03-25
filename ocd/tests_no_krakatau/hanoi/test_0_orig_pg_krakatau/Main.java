class Main {
    private static int a;
    private static java.util.Stack b;
    private static java.util.Stack c;
    private static java.util.Stack d;
    private static int e;
    private static int f;
    private static int g;
    
    Main()
    {
        super();
    }
    
    private static void a(java.util.Stack a0, java.util.Stack a1, java.util.Stack a2)
    {
        a1.push(a0.pop());
        a = a + 1;
        Main.a();
        a2.push(a0.pop());
        a = a + 1;
        Main.a();
        a2.push(a1.pop());
        a = a + 1;
        Main.a();
    }
    
    private static int a(int i, java.util.Stack a0, java.util.Stack a1, java.util.Stack a2)
    {
        if (i > 4)
        {
            Main.a(i - 2, a0, a1, a2);
            a1.push(a0.pop());
            a = a + 1;
            Main.a();
            Main.a(i - 2, a2, a0, a1);
            a2.push(a0.pop());
            a = a + 1;
            Main.a();
            Main.a(i - 1, a1, a0, a2);
            return 1;
        }
        if (i % 2 == 0)
        {
            Main.a(a0, a1, a2);
            int i0 = i + -1;
            if (i0 == 1)
            {
                return 1;
            }
            a1.push(a0.pop());
            a = a + 1;
            Main.a();
            Main.a(a2, a0, a1);
            a2.push(a0.pop());
            a = a + 1;
            Main.a();
            Main.a(i0, a1, a0, a2);
        }
        else
        {
            if (i == 1)
            {
                return -1;
            }
            Main.a(a0, a2, a1);
            a2.push(a0.pop());
            a = a + 1;
            Main.a();
            Main.a(a1, a0, a2);
        }
        return 1;
    }
    
    private static void a()
    {
        int i = e;
        int i0 = b.size();
        label3: {
            label0: {
                label2: {
                    if (i == i0)
                    {
                        break label2;
                    }
                    break label0;
                }
                int i1 = f;
                int i2 = c.size();
                label1: {
                    if (i1 == i2)
                    {
                        break label1;
                    }
                    break label0;
                }
                if (g != d.size())
                {
                    break label0;
                }
                break label3;
            }
            int i3 = b.size() - e;
            int i4 = c.size() - f;
            d.size();
            int dummy = g;
            if (i3 == 1)
            {
                if (i4 == -1)
                {
                    System.out.print(new StringBuilder("Move Disc ").append(b.peek()).append(" From B To A").toString());
                }
                else
                {
                    System.out.print(new StringBuilder("Move Disc ").append(b.peek()).append(" From C To A").toString());
                }
            }
            else if (i4 == 1)
            {
                if (i3 == -1)
                {
                    System.out.print(new StringBuilder("Move Disc ").append(c.peek()).append(" From A To B").toString());
                }
                else
                {
                    System.out.print(new StringBuilder("Move Disc ").append(c.peek()).append(" From C To B").toString());
                }
            }
            else if (i3 == -1)
            {
                System.out.print(new StringBuilder("Move Disc ").append(d.peek()).append(" From A To C").toString());
            }
            else
            {
                System.out.print(new StringBuilder("Move Disc ").append(d.peek()).append(" From B To C").toString());
            }
            e = b.size();
            f = c.size();
            g = d.size();
            System.out.println();
        }
        Main.a(b);
        System.out.print(" , ");
        Main.a(c);
        System.out.print(" , ");
        Main.a(d);
        System.out.print(" , ");
    }
    
    private static void a(java.util.Stack a0)
    {
        System.out.print(a0.toString());
    }
    
    public static void main(String[] a0)
    {
        try
        {
            while(true)
            {
                System.out.print("\nEnter the number of discs (-1 to exit): ");
                String s = new java.io.BufferedReader((java.io.Reader)new java.io.InputStreamReader(System.in)).readLine();
                a = 0;
                int i = Integer.parseInt(s);
                if (i == -1)
                {
                    break;
                }
                {
                    if (i > 1 && i < 10)
                    {
                        int i0 = i;
                        while(i0 > 0)
                        {
                            b.push((Object)Integer.valueOf(i0));
                            i0 = i0 + -1;
                        }
                        e = b.size();
                        f = c.size();
                        g = d.size();
                        Main.a();
                        Main.a(i, b, c, d);
                        System.out.println(new StringBuilder("Total Moves = ").append(a).toString());
                        while(d.size() > 0)
                        {
                            d.pop();
                        }
                        continue;
                    }
                    System.out.println("Enter between 2 - 9");
                }
            }
            System.out.println("Good Bye!");
        }
        catch(Exception a1)
        {
            a1.printStackTrace();
            return;
        }
    }
    
    static
    {
        a = 0;
        b = new java.util.Stack();
        c = new java.util.Stack();
        d = new java.util.Stack();
        e = 0;
        f = 0;
        g = 0;
    }
}
